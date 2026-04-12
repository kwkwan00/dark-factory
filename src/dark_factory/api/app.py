"""FastAPI application for AI Dark Factory."""

from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

log = structlog.get_logger()


def _validate_configured_models(settings) -> None:
    """M6: best-effort startup validation of configured model names.

    Queries the Anthropic models endpoint (if a key is present) and
    checks that ``settings.llm.model`` + ``settings.evaluation.eval_model``
    appear in the returned catalogue. Logs a WARN on any mismatch —
    never raises — because:

    - The check is optional (keys may not be set at startup, e.g.
      during unit tests or when keys are per-request overrides).
    - Operators may be running on a model that's newer than the
      catalogue's last refresh.
    - A provider outage during lifespan must not block app startup.
    """
    import os

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        log.debug("model_validation_skipped", reason="no_anthropic_key")
        return

    try:
        from dark_factory.api.routes_models import _fetch_anthropic_models

        models = _fetch_anthropic_models(anthropic_key)
    except Exception as exc:
        log.debug(
            "model_validation_failed_fetch",
            error=str(exc)[:200],
        )
        return

    available = {m.get("id", "") for m in models if isinstance(m, dict)}
    if not available:
        log.debug("model_validation_skipped", reason="empty_catalogue")
        return

    configured_llm = settings.llm.model
    if configured_llm and configured_llm not in available:
        log.warning(
            "configured_llm_model_unknown",
            model=configured_llm,
            hint=(
                "Configured anthropic model does not appear in the "
                "provider's catalogue. This may be a typo, a "
                "deprecated model, or a newer model the catalogue "
                "hasn't listed yet. Pipeline runs will fail at the "
                "first LLM call if the name is invalid."
            ),
        )

    configured_eval = getattr(settings.evaluation, "eval_model", "")
    # Eval model is typically OpenAI (gpt-*), so skipping Anthropic
    # validation for it is correct. A dedicated OpenAI catalogue
    # fetch would be symmetric but adds complexity for little value.
    if configured_eval and configured_eval.startswith("claude"):
        if configured_eval not in available:
            log.warning(
                "configured_eval_model_unknown",
                model=configured_eval,
                hint=(
                    "Configured eval_model starts with 'claude' but "
                    "is not in the Anthropic catalogue. If it's an "
                    "OpenAI model, this check is safely skipped — "
                    "otherwise the eval calls will fail."
                ),
            )


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup; tear down on shutdown."""
    from dark_factory.config import load_settings
    from dark_factory.graph.client import Neo4jClient
    from dark_factory.log import setup_logging

    settings = load_settings()
    setup_logging(level=settings.logging.level, fmt=settings.logging.format)
    app.state.settings = settings

    # Shared Neo4j client for graph queries (connection pool managed by driver)
    app.state.neo4j_client = Neo4jClient(settings.neo4j)

    # Shared vector repository (optional, used by memory + spec/code search)
    app.state.vector_repo = None
    if settings.qdrant.enabled:
        try:
            from dark_factory.vector.client import QdrantClientWrapper
            from dark_factory.vector.collections import ensure_collections
            from dark_factory.vector.embeddings import EmbeddingService
            from dark_factory.vector.repository import VectorRepository

            qdrant_client = QdrantClientWrapper(settings.qdrant)
            ensure_collections(qdrant_client)
            app.state.vector_repo = VectorRepository(
                client=qdrant_client,
                embeddings=EmbeddingService(model=settings.qdrant.embedding_model),
            )
            # L6 fix: ping Qdrant to confirm it's actually reachable
            # before declaring the vector repo initialised. Previously
            # we'd install a VectorRepository that would silently fail
            # every search on the first real query, and operators
            # wouldn't know until recall quality degraded. Now the
            # startup log reflects whether the backing store is
            # actually up. Best-effort — the ping failure downgrades
            # to a WARN, not an ERROR, because vector search is
            # optional; the app still boots.
            try:
                # Minimal reachability check: list collections is the
                # lightest Qdrant op that exercises the HTTP path.
                qdrant_client.client.get_collections()
                log.info("vector_repo_initialized", qdrant_url=settings.qdrant.url)
            except Exception as ping_exc:
                log.warning(
                    "vector_repo_unreachable_at_startup",
                    qdrant_url=settings.qdrant.url,
                    error=str(ping_exc),
                )
        except Exception as exc:
            log.warning("vector_repo_init_failed", error=str(exc))
            app.state.vector_repo = None

    # Shared memory repository (optional). H5 fix: pass vector_repo so the
    # dashboard endpoints see the same hybrid (Neo4j + Qdrant) results that
    # the swarm agents see.
    if settings.memory.enabled:
        from dark_factory.config import Neo4jConfig
        from dark_factory.memory.repository import MemoryRepository
        from dark_factory.memory.schema import init_memory_schema

        mem_config = Neo4jConfig(
            uri=settings.neo4j.uri,
            database=settings.memory.database,
            user=settings.neo4j.user,
            password=settings.neo4j.password,
        )
        memory_client = Neo4jClient(mem_config)
        init_memory_schema(memory_client)
        app.state.memory_repo = MemoryRepository(
            memory_client,
            vector_repo=app.state.vector_repo,
            dedup_threshold=settings.pipeline.memory_dedup_threshold,
        )
        app.state.memory_client = memory_client

        # H5/H6 fix: install the shared memory_repo / vector_repo into the
        # tools module so init_swarm_context can reuse them instead of
        # creating its own duplicate driver pool per pipeline run.
        from dark_factory.agents.tools import set_memory_repo, set_vector_repo

        set_memory_repo(app.state.memory_repo)
        if app.state.vector_repo is not None:
            set_vector_repo(app.state.vector_repo)
    else:
        app.state.memory_repo = None
        app.state.memory_client = None

    app.state.watcher = None

    # Progress broker: fans out swarm events to the Run tab and the Agent Logs tab
    import asyncio as _asyncio

    from dark_factory.agents.progress import ProgressBroker
    from dark_factory.agents.tools import set_progress_broker

    # Each subscriber captures its own loop (see progress.py) — the broker
    # itself is loop-agnostic.
    broker = ProgressBroker()
    app.state.progress_broker = broker
    set_progress_broker(broker)

    # Metrics recorder (Postgres-backed). Best-effort — if Postgres is
    # disabled or unreachable, the factory returns (None, None) and the
    # rest of the app keeps working without metrics.
    from dark_factory.agents.tools import set_metrics_recorder
    from dark_factory.metrics.recorder import build_recorder_from_settings

    metrics_recorder, metrics_client = build_recorder_from_settings(settings)
    app.state.metrics_recorder = metrics_recorder
    app.state.metrics_client = metrics_client
    if metrics_recorder is not None:
        set_metrics_recorder(metrics_recorder)

    # BackgroundLoop periodic sampler — always on. Writes to both Prometheus
    # (always) and the Postgres recorder (when enabled). The sampler also
    # polls the progress broker and metrics recorder gauges.
    from dark_factory.metrics.bg_sampler import BackgroundLoopSampler

    sampler = BackgroundLoopSampler(interval_seconds=10.0)
    sampler.start()
    app.state.bg_loop_sampler = sampler

    # C1 fix: serialize pipeline runs. Concurrent runs would clobber
    # _current_run_id / _current_feature module-globals in tools.py.
    app.state.run_lock = _asyncio.Lock()

    # M6 fix: validate the configured LLM model names against the
    # provider's catalogue at startup. Logs a WARN for unknown
    # models (typo, deprecated) but never hard-fails — operators
    # may be running on a newer model that the list hasn't caught
    # up with yet. The cost is one HTTP round-trip to the provider
    # during lifespan; we guard it tightly so a provider outage
    # can't block app startup.
    _validate_configured_models(settings)

    log.info("app_started", model=settings.llm.model)
    yield

    # Shutdown
    set_progress_broker(None)
    set_metrics_recorder(None)
    from dark_factory.agents.tools import set_memory_repo, set_vector_repo

    set_memory_repo(None)
    set_vector_repo(None)
    if app.state.bg_loop_sampler is not None:
        app.state.bg_loop_sampler.stop()
    if metrics_recorder is not None:
        metrics_recorder.close()
    if metrics_client is not None:
        metrics_client.close()
    app.state.neo4j_client.close()
    if app.state.memory_client:
        app.state.memory_client.close()
    watcher = app.state.watcher
    if watcher and watcher.is_running:
        watcher.stop()

    # C3 fix: cleanly stop the background event loop so pending Claude
    # SDK subprocess cleanup tasks have a chance to complete.
    from dark_factory.agents.background_loop import BackgroundLoop

    BackgroundLoop.reset()
    log.info("app_stopped")


# ── App ────────────────────────────────────────────────────────────────────────


app = FastAPI(title="AI Dark Factory", version="0.1.0", lifespan=lifespan)

# C1 fix: allow_credentials removed — wildcard origin + credentials is invalid
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom 422 handler that strips the raw request ``body`` from the
    response payload.

    CRITICAL security fix: FastAPI's default handler returns
    ``{"detail": exc.errors(), "body": exc.body}`` — meaning the entire
    raw request body is echoed back to the client on any validation
    error. For the ``/api/agent/run`` endpoint this would leak the
    ``anthropic_api_key`` / ``openai_api_key`` fields from the request
    into the 422 response, which the frontend's ``describeFailure()``
    would then write into ``Error.message`` and subsequently into browser
    console logs.

    Combined with ``hide_input_in_errors=True`` on the ``RunRequest``
    model (which suppresses the ``input`` field in each individual
    ``errors()`` entry), this handler completely cuts off the leak path:
    the client gets the validation messages, never the submitted values.
    """
    # Scrub any residual ``input`` / ``ctx`` leakage: even with
    # ``hide_input_in_errors=True`` set on the model, ``ctx`` can contain
    # raw ``ValueError`` instances whose str() repr may echo the value.
    # jsonable_encoder converts them to safe dicts; we then strip ``input``
    # and ``ctx`` defensively so no raw request data escapes.
    errors = jsonable_encoder(exc.errors())
    scrubbed = []
    for err in errors:
        if isinstance(err, dict):
            err.pop("input", None)
            err.pop("ctx", None)
        scrubbed.append(err)
    return JSONResponse(
        status_code=422,
        content={"detail": scrubbed},
    )

from dark_factory.api.routes_admin import router as admin_router
from dark_factory.api.routes_agent import router as agent_router
from dark_factory.api.routes_dashboard import router as dashboard_router
from dark_factory.api.routes_metrics import router as metrics_router
from dark_factory.api.routes_models import router as models_router
from dark_factory.api.routes_runs import router as runs_router
from dark_factory.api.routes_upload import router as upload_router

app.include_router(agent_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(runs_router, prefix="/api")
app.include_router(models_router, prefix="/api")


@app.get("/health")
async def health():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus scrape endpoint (text exposition format).

    Always on — prometheus_client collectors are in-memory counters with no
    I/O, so there's no need to gate this behind a config flag. The Postgres
    metrics store remains optional independently.
    """
    from dark_factory.metrics.prometheus import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Serve React SPA (only when built) ──────────────────────────────────────────

STATIC_DIR = pathlib.Path(__file__).parent / "static"
if STATIC_DIR.exists():
    _assets = STATIC_DIR / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # L6 fix: don't swallow mistyped /api/* paths — return 404 for those
        if full_path.startswith("api/") or full_path == "api":
            return JSONResponse({"error": "Not found"}, status_code=404)
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse(
            {"error": "Frontend not built. Run: cd frontend && npm run build"},
            status_code=404,
        )
