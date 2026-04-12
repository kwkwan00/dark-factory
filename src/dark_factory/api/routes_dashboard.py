"""REST dashboard endpoints — gap finder, history, memory, eval, settings, health, watcher."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from dark_factory.config import Settings

import structlog
from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

log = structlog.get_logger()

router = APIRouter()


# ── Graph Gap Finder ───────────────────────────────────────────────────────────


@router.get("/graph/gaps")
def get_graph_gaps(
    request: Request,
    stale_days: int = Query(
        default=7,
        ge=1,
        le=365,
        description=(
            "A requirement is 'stale' when all of its implementing specs' "
            "most recent eval is older than this many days."
        ),
    ),
):
    """Return actionable gaps in the knowledge graph.

    Four categories:

    1. **unplanned_requirements** — Requirements with no ``IMPLEMENTS``
       spec. The planner either missed them or they're new.
    2. **specs_without_artifacts** — Specs that have no rows in the
       Postgres ``artifact_writes`` table. Codegen failed to produce
       anything for them.
    3. **specs_failing_evals** — Specs whose latest ``eval_metrics`` row
       has ``passed=false``. Quality gap.
    4. **stale_requirements** — Requirements whose implementing specs
       haven't been re-evaluated in the last ``stale_days`` days.

    Categories 2–4 require the Postgres metrics store. When Postgres is
    disabled, the endpoint still returns category 1 (Neo4j-only) and
    sets ``enabled_postgres: false`` so the UI can show a partial view.
    """
    from datetime import datetime, timedelta, timezone

    neo4j_client = request.app.state.neo4j_client
    metrics_client = getattr(request.app.state, "metrics_client", None)

    # ── Pass 1: pull requirements, specs, and IMPLEMENTS from Neo4j ──
    try:
        with neo4j_client.session() as session:
            # Unplanned: no incoming IMPLEMENTS edge. Include source_file
            # so the UI can link back to where the requirement came from.
            unplanned = [
                dict(r)
                for r in session.run(
                    """
                    MATCH (r:Requirement)
                    WHERE NOT (r)<-[:IMPLEMENTS]-(:Spec)
                    RETURN r.id AS id,
                           r.title AS title,
                           r.priority AS priority,
                           r.source_file AS source_file
                    ORDER BY r.priority, r.id
                    """
                )
            ]

            # All specs + their IMPLEMENTS target(s). We need this even
            # when Postgres is disabled so the "specs without artifacts"
            # section can at least render an empty state keyed by the
            # full spec list.
            specs = [
                dict(r)
                for r in session.run(
                    """
                    MATCH (s:Spec)
                    OPTIONAL MATCH (s)-[:IMPLEMENTS]->(r:Requirement)
                    RETURN s.id AS id,
                           s.title AS title,
                           s.capability AS capability,
                           collect(DISTINCT r.id) AS requirement_ids
                    ORDER BY s.id
                    """
                )
            ]

            # Spec → requirement inverse map, used for the stale
            # requirement rollup below.
            spec_to_reqs: dict[str, list[str]] = {
                s["id"]: [rid for rid in (s.get("requirement_ids") or []) if rid]
                for s in specs
            }
    except Exception as exc:
        log.warning("graph_gaps_neo4j_failed", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=f"Knowledge graph unavailable: {exc}",
        ) from exc

    payload: dict = {
        "enabled_postgres": metrics_client is not None,
        "stale_days": stale_days,
        "unplanned_requirements": unplanned,
        "specs_without_artifacts": [],
        "specs_failing_evals": [],
        "stale_requirements": [],
        "totals": {
            "requirements": 0,  # filled in below
            "specs": len(specs),
        },
    }

    # Count requirements in the graph too — for the tab header summary.
    try:
        with neo4j_client.session() as session:
            row = session.run("MATCH (r:Requirement) RETURN count(r) AS cnt").single()
            payload["totals"]["requirements"] = int(row["cnt"] if row else 0)
    except Exception:  # pragma: no cover — not critical
        pass

    if metrics_client is None:
        return payload

    # ── Pass 2: cross-reference with Postgres metrics ────────────────
    try:
        with metrics_client.connection() as conn:
            with conn.cursor() as cur:
                # Spec IDs that have at least one artifact row.
                cur.execute(
                    """
                    SELECT DISTINCT spec_id
                    FROM artifact_writes
                    WHERE spec_id IS NOT NULL AND spec_id <> ''
                    """
                )
                specs_with_artifacts: set[str] = {
                    row["spec_id"] for row in cur.fetchall()
                }

                # Latest eval row per spec. DISTINCT ON is Postgres-
                # specific and relies on the ORDER BY matching.
                cur.execute(
                    """
                    SELECT DISTINCT ON (spec_id)
                           spec_id,
                           metric_name,
                           score,
                           passed,
                           timestamp
                    FROM eval_metrics
                    WHERE spec_id IS NOT NULL AND spec_id <> ''
                    ORDER BY spec_id, timestamp DESC
                    """
                )
                latest_eval_by_spec: dict[str, dict] = {
                    row["spec_id"]: {
                        "metric_name": row["metric_name"],
                        "score": row["score"],
                        "passed": row["passed"],
                        "timestamp": (
                            row["timestamp"].isoformat()
                            if row["timestamp"]
                            else None
                        ),
                        "timestamp_raw": row["timestamp"],
                    }
                    for row in cur.fetchall()
                }
    except Exception as exc:
        log.warning("graph_gaps_postgres_failed", error=str(exc))
        # Return whatever we have from Neo4j + mark postgres as failed.
        payload["enabled_postgres"] = False
        payload["postgres_error"] = str(exc)
        return payload

    # ── Compose the three Postgres-backed gap lists ─────────────────
    # 2. Specs without any generated artifact
    payload["specs_without_artifacts"] = [
        {
            "id": s["id"],
            "title": s.get("title"),
            "capability": s.get("capability"),
            "requirement_ids": spec_to_reqs.get(s["id"], []),
        }
        for s in specs
        if s["id"] not in specs_with_artifacts
    ]

    # 3. Specs whose most recent eval did not pass
    payload["specs_failing_evals"] = [
        {
            "id": s["id"],
            "title": s.get("title"),
            "capability": s.get("capability"),
            "requirement_ids": spec_to_reqs.get(s["id"], []),
            "metric_name": latest_eval_by_spec[s["id"]]["metric_name"],
            "score": latest_eval_by_spec[s["id"]]["score"],
            "last_eval_at": latest_eval_by_spec[s["id"]]["timestamp"],
        }
        for s in specs
        if s["id"] in latest_eval_by_spec
        and not latest_eval_by_spec[s["id"]]["passed"]
    ]

    # 4. Stale requirements: every implementing spec's last eval is
    #    older than `stale_days` ago (or has no eval at all).
    #
    #    This is naturally a secondary rollup: we invert spec_to_reqs
    #    into req_to_specs, then for each requirement check whether
    #    ALL its specs' latest evals are stale.
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=stale_days)
    req_to_specs: dict[str, list[str]] = {}
    for spec_id, req_ids in spec_to_reqs.items():
        for rid in req_ids:
            req_to_specs.setdefault(rid, []).append(spec_id)

    stale: list[dict] = []
    for rid, spec_ids in req_to_specs.items():
        if not spec_ids:
            continue
        newest_ts = None
        for sid in spec_ids:
            ev = latest_eval_by_spec.get(sid)
            if ev and ev["timestamp_raw"] is not None:
                ts = ev["timestamp_raw"]
                if newest_ts is None or ts > newest_ts:
                    newest_ts = ts
        if newest_ts is None or newest_ts < cutoff:
            stale.append(
                {
                    "id": rid,
                    "spec_count": len(spec_ids),
                    "last_eval_at": (
                        newest_ts.isoformat() if newest_ts else None
                    ),
                }
            )

    # Sort stale by oldest-first so the UI shows the most stale first.
    stale.sort(key=lambda r: r.get("last_eval_at") or "")
    payload["stale_requirements"] = stale

    return payload


# ── Run History ────────────────────────────────────────────────────────────────


@router.get("/history")
def get_history(request: Request, limit: int = Query(default=20, ge=1, le=100)):
    """Return recent pipeline run history."""
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        return {"runs": [], "message": "Memory system is disabled"}
    runs = memory_repo.get_run_history(limit=limit)
    return {"runs": runs}


# ── Memory Search ──────────────────────────────────────────────────────────────


@router.get("/memory/list")
def list_memory(
    request: Request,
    memory_type: Literal["all", "pattern", "mistake", "solution", "strategy"] = Query(
        default="all", alias="type"
    ),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Browse all procedural memories ordered by relevance.

    Unlike ``/memory/search``, this endpoint requires no keywords — it
    returns the most relevant N memories so the dashboard can show what's
    available without forcing the user to guess search terms.
    """
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        return {"results": [], "message": "Memory system is disabled", "total": 0}
    results = memory_repo.list_memories(memory_type=memory_type, limit=limit)
    return {"results": results, "total": len(results), "type": memory_type}


@router.get("/memory/search")
def search_memory(
    request: Request,
    keywords: str = Query(..., min_length=1),
    # H5/H7 fix: renamed from 'type' (shadows builtin), constrained to valid values
    memory_type: Literal["all", "pattern", "mistake", "solution", "strategy"] = Query(
        default="all", alias="type"
    ),
):
    """Search procedural memory by keywords."""
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        return {"results": [], "message": "Memory system is disabled"}
    results = []
    search_map = {
        "pattern": memory_repo.search_patterns,
        "mistake": memory_repo.search_mistakes,
        "solution": memory_repo.search_solutions,
        "strategy": memory_repo.get_strategies,
    }
    targets = (
        list(search_map.items())
        if memory_type == "all"
        else [(memory_type, search_map[memory_type])]
    )
    for mtype, fn in targets:
        found = fn(keywords=keywords)
        results.extend([{"type": mtype, **m} for m in found])
    return {"results": results, "keywords": keywords}


# ── Eval Scores ────────────────────────────────────────────────────────────────


@router.get("/eval")
def list_evals(
    request: Request,
    run_limit: int = Query(default=20, ge=1, le=100),
    run_id: str | None = Query(
        default=None,
        pattern=r"^[A-Za-z0-9_\-]{1,128}$",
        description=(
            "When set, filter the response to a single run by id. "
            "Used by the Run Detail popup's Evaluations screen so it "
            "doesn't have to fetch every run's evals just to display one."
        ),
    ),
):
    """Browse all eval results grouped by pipeline run.

    Returns a hierarchical structure of run → spec → attempts. With
    no ``run_id`` filter, returns the most recent ``run_limit`` runs.
    With a ``run_id``, returns just that one run (still wrapped in a
    ``runs: [...]`` array for response-shape consistency) — or an
    empty array if the run isn't in memory.
    """
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        return {"runs": [], "message": "Memory system is disabled"}
    runs = memory_repo.list_evals_by_run(run_limit=run_limit)
    if run_id is not None:
        runs = [r for r in runs if r.get("run_id") == run_id]
    return {"runs": runs}


@router.get("/eval/{spec_id}")
def get_eval_history(
    request: Request,
    # C3 fix: regex constraint to prevent injection
    spec_id: str = Path(pattern=r"^[a-zA-Z0-9_. -]+$"),
    limit: int = Query(default=10, ge=1, le=100),
):
    """Return eval score history for a spec."""
    memory_repo = request.app.state.memory_repo
    if memory_repo is None:
        return {"history": [], "message": "Memory system is disabled"}
    history = memory_repo.get_eval_history(spec_id=spec_id, limit=limit)
    return {"spec_id": spec_id, "history": history}


# ── Tunable Pipeline Settings ─────────────────────────────────────────────────


class PipelineSettingsResponse(BaseModel):
    """Tunable settings exposed by the Settings tab."""

    max_parallel_features: int
    max_parallel_specs: int
    max_spec_handoffs: int
    max_codegen_handoffs: int
    spec_eval_threshold: float
    enable_spec_decomposition: bool
    max_specs_per_requirement: int
    reuse_existing_specs: bool
    max_reconciliation_turns: int
    reconciliation_timeout_seconds: int
    requirement_dedup_threshold: float
    enable_e2e_validation: bool
    max_e2e_turns: int
    e2e_timeout_seconds: int
    e2e_browsers: list[str]
    enable_episodic_memory: bool
    memory_dedup_threshold: float

    # Editable model selections
    llm_model: str
    eval_model: str

    # Read-only metadata
    output_dir: str


class PipelineSettingsUpdate(BaseModel):
    """Partial update — only fields present in the request body are applied."""

    max_parallel_features: int | None = Field(default=None, ge=1, le=8)
    max_parallel_specs: int | None = Field(default=None, ge=1, le=8)
    max_spec_handoffs: int | None = Field(default=None, ge=1, le=10)
    max_codegen_handoffs: int | None = Field(default=None, ge=5, le=100)
    spec_eval_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    enable_spec_decomposition: bool | None = None
    max_specs_per_requirement: int | None = Field(default=None, ge=1, le=32)
    reuse_existing_specs: bool | None = None
    max_reconciliation_turns: int | None = Field(default=None, ge=1, le=500)
    reconciliation_timeout_seconds: int | None = Field(
        default=None, ge=60, le=7200
    )
    requirement_dedup_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    enable_e2e_validation: bool | None = None
    max_e2e_turns: int | None = Field(default=None, ge=1, le=500)
    e2e_timeout_seconds: int | None = Field(default=None, ge=60, le=7200)
    e2e_browsers: list[str] | None = Field(default=None, min_length=1)
    enable_episodic_memory: bool | None = None
    memory_dedup_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    # Model name changes are validated as non-empty strings. We don't
    # enum-restrict them here — the frontend offers a curated dropdown
    # but operators can still pipe in any string via the API / a custom
    # override.
    llm_model: str | None = Field(default=None, min_length=1, max_length=128)
    eval_model: str | None = Field(default=None, min_length=1, max_length=128)


class _SettingsPayloadDict(TypedDict):
    """L3 fix: typed shape for the settings GET/PATCH responses.

    Mirrors :class:`PipelineSettingsResponse` field-for-field so
    the dict returned by :func:`_settings_payload` cannot drift
    silently from the Pydantic schema the REST clients consume.
    Any new field added to ``PipelineSettingsResponse`` should
    also be added here; a TypeError will fire if they diverge.
    """

    max_parallel_features: int
    max_parallel_specs: int
    max_spec_handoffs: int
    max_codegen_handoffs: int
    spec_eval_threshold: float
    enable_spec_decomposition: bool
    max_specs_per_requirement: int
    reuse_existing_specs: bool
    max_reconciliation_turns: int
    reconciliation_timeout_seconds: int
    requirement_dedup_threshold: float
    enable_e2e_validation: bool
    max_e2e_turns: int
    e2e_timeout_seconds: int
    e2e_browsers: list[str]
    enable_episodic_memory: bool
    memory_dedup_threshold: float
    output_dir: str
    llm_model: str
    eval_model: str


def _settings_payload(settings: "Settings") -> _SettingsPayloadDict:
    # L11 fix: single source of truth for the eval model. Read via the
    # metrics-module getter so runtime Settings-tab changes are reflected
    # immediately (the old ``from ... import EVAL_MODEL`` pattern captured
    # the module global at import time and stayed stale across PATCHes).
    from dark_factory.evaluation.metrics import get_eval_model

    p = settings.pipeline
    return {
        "max_parallel_features": p.max_parallel_features,
        "max_parallel_specs": p.max_parallel_specs,
        "max_spec_handoffs": p.max_spec_handoffs,
        "max_codegen_handoffs": p.max_codegen_handoffs,
        "spec_eval_threshold": p.spec_eval_threshold,
        "enable_spec_decomposition": p.enable_spec_decomposition,
        "max_specs_per_requirement": p.max_specs_per_requirement,
        "reuse_existing_specs": p.reuse_existing_specs,
        "max_reconciliation_turns": p.max_reconciliation_turns,
        "reconciliation_timeout_seconds": p.reconciliation_timeout_seconds,
        "requirement_dedup_threshold": p.requirement_dedup_threshold,
        "enable_e2e_validation": p.enable_e2e_validation,
        "max_e2e_turns": p.max_e2e_turns,
        "e2e_timeout_seconds": p.e2e_timeout_seconds,
        "e2e_browsers": list(p.e2e_browsers),
        "enable_episodic_memory": p.enable_episodic_memory,
        "memory_dedup_threshold": p.memory_dedup_threshold,
        "output_dir": p.output_dir,
        "llm_model": settings.llm.model,
        "eval_model": get_eval_model(),
    }


@router.get("/settings", response_model=PipelineSettingsResponse)
def get_pipeline_settings(request: Request):
    """Return the current tunable pipeline settings."""
    return _settings_payload(request.app.state.settings)


@router.patch("/settings", response_model=PipelineSettingsResponse)
def update_pipeline_settings(request: Request, body: PipelineSettingsUpdate):
    """Update tunable pipeline settings at runtime.

    Only fields present in the request body are applied. Changes take effect
    on the **next** pipeline run — currently-running pipelines are not interrupted.

    Returns the new full settings payload after applying changes.
    """
    settings_obj = request.app.state.settings
    pipeline = settings_obj.pipeline
    updates: dict[str, object] = {}

    try:
        if body.max_parallel_features is not None:
            pipeline.max_parallel_features = body.max_parallel_features
            updates["max_parallel_features"] = body.max_parallel_features
        if body.max_parallel_specs is not None:
            pipeline.max_parallel_specs = body.max_parallel_specs
            updates["max_parallel_specs"] = body.max_parallel_specs
        if body.max_spec_handoffs is not None:
            pipeline.max_spec_handoffs = body.max_spec_handoffs
            updates["max_spec_handoffs"] = body.max_spec_handoffs
        if body.max_codegen_handoffs is not None:
            pipeline.max_codegen_handoffs = body.max_codegen_handoffs
            updates["max_codegen_handoffs"] = body.max_codegen_handoffs
        if body.spec_eval_threshold is not None:
            pipeline.spec_eval_threshold = body.spec_eval_threshold
            updates["spec_eval_threshold"] = body.spec_eval_threshold
        if body.enable_spec_decomposition is not None:
            pipeline.enable_spec_decomposition = body.enable_spec_decomposition
            updates["enable_spec_decomposition"] = body.enable_spec_decomposition
        if body.max_specs_per_requirement is not None:
            pipeline.max_specs_per_requirement = body.max_specs_per_requirement
            updates["max_specs_per_requirement"] = body.max_specs_per_requirement
        if body.reuse_existing_specs is not None:
            pipeline.reuse_existing_specs = body.reuse_existing_specs
            updates["reuse_existing_specs"] = body.reuse_existing_specs
        if body.max_reconciliation_turns is not None:
            pipeline.max_reconciliation_turns = body.max_reconciliation_turns
            updates["max_reconciliation_turns"] = body.max_reconciliation_turns
        if body.reconciliation_timeout_seconds is not None:
            pipeline.reconciliation_timeout_seconds = (
                body.reconciliation_timeout_seconds
            )
            updates["reconciliation_timeout_seconds"] = (
                body.reconciliation_timeout_seconds
            )
        if body.requirement_dedup_threshold is not None:
            pipeline.requirement_dedup_threshold = (
                body.requirement_dedup_threshold
            )
            updates["requirement_dedup_threshold"] = (
                body.requirement_dedup_threshold
            )
        if body.enable_e2e_validation is not None:
            pipeline.enable_e2e_validation = body.enable_e2e_validation
            updates["enable_e2e_validation"] = body.enable_e2e_validation
        if body.max_e2e_turns is not None:
            pipeline.max_e2e_turns = body.max_e2e_turns
            updates["max_e2e_turns"] = body.max_e2e_turns
        if body.e2e_timeout_seconds is not None:
            pipeline.e2e_timeout_seconds = body.e2e_timeout_seconds
            updates["e2e_timeout_seconds"] = body.e2e_timeout_seconds
        if body.e2e_browsers is not None:
            # Validate against the allowed engine set so an operator
            # can't persist a malformed browser list through the UI
            # that would then crash the next run's Playwright config.
            allowed = {"chromium", "firefox", "webkit"}
            cleaned = [
                b.strip().lower() for b in body.e2e_browsers if b and b.strip()
            ]
            invalid = [b for b in cleaned if b not in allowed]
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid e2e_browsers entries: {invalid}. "
                        f"Allowed: {sorted(allowed)}"
                    ),
                )
            if not cleaned:
                raise HTTPException(
                    status_code=400,
                    detail="e2e_browsers must contain at least one browser",
                )
            pipeline.e2e_browsers = cleaned
            updates["e2e_browsers"] = cleaned
        if body.enable_episodic_memory is not None:
            pipeline.enable_episodic_memory = body.enable_episodic_memory
            updates["enable_episodic_memory"] = body.enable_episodic_memory
        if body.memory_dedup_threshold is not None:
            pipeline.memory_dedup_threshold = body.memory_dedup_threshold
            updates["memory_dedup_threshold"] = body.memory_dedup_threshold
            # Live-update the installed MemoryRepository so the new
            # threshold takes effect immediately rather than on next
            # app start. Tolerate a missing repo (memory disabled).
            repo = getattr(request.app.state, "memory_repo", None)
            if repo is not None:
                try:
                    repo.set_dedup_threshold(body.memory_dedup_threshold)
                except Exception as exc:  # pragma: no cover — defensive
                    log.warning("memory_dedup_threshold_live_update_failed", error=str(exc))
        if body.llm_model is not None:
            settings_obj.llm.model = body.llm_model.strip()
            updates["llm_model"] = settings_obj.llm.model
        if body.eval_model is not None:
            new_eval = body.eval_model.strip()
            settings_obj.evaluation.eval_model = new_eval
            # Propagate to the metrics module global so the NEXT pipeline
            # run's GEval builders pick up the new model. Already-built
            # metrics in flight are unaffected (we explicitly document
            # that changes take effect on the next run).
            from dark_factory.evaluation.metrics import set_eval_model

            set_eval_model(new_eval)
            updates["eval_model"] = new_eval
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log.info("pipeline_settings_updated", **updates)
    return _settings_payload(request.app.state.settings)


# ── Health ─────────────────────────────────────────────────────────────────────


@router.get("/health")
def get_health(request: Request):
    """Return health status of all external services."""
    from dark_factory.ui.health import check_all

    settings = request.app.state.settings
    # M14 fix: reuse the shared neo4j_client / metrics_client instead of
    # opening new ones per request.
    raw = check_all(
        settings,
        neo4j_client=request.app.state.neo4j_client,
        metrics_client=getattr(request.app.state, "metrics_client", None),
    )
    return {
        service: {"ok": ok, "message": message}
        for service, (ok, message) in raw.items()
    }


# ── File Watcher ───────────────────────────────────────────────────────────────


@router.post("/watch/start")
def watch_start(request: Request):
    """Start the file watcher."""
    from dark_factory.ui.watcher import FileWatcher

    settings = request.app.state.settings
    watcher = request.app.state.watcher
    if watcher and watcher.is_running:
        return {"status": "already_running", "paths": watcher.paths}

    watcher = FileWatcher(
        paths=settings.watch.paths,
        debounce_seconds=settings.watch.debounce_seconds,
    )
    watcher.start()
    request.app.state.watcher = watcher
    log.info("watcher_started_via_api", paths=settings.watch.paths)
    return {"status": "started", "paths": settings.watch.paths}


@router.post("/watch/stop")
def watch_stop(request: Request):
    """Stop the file watcher."""
    watcher = request.app.state.watcher
    if watcher and watcher.is_running:
        watcher.stop()
        request.app.state.watcher = None
        return {"status": "stopped"}
    return {"status": "not_running"}


@router.get("/watch/status")
def watch_status(request: Request):
    """Return current watcher status."""
    watcher = request.app.state.watcher
    if watcher and watcher.is_running:
        last = watcher.last_event
        return {
            "running": True,
            "paths": watcher.paths,
            "last_event": (
                {"path": last.path, "type": last.event_type, "timestamp": last.timestamp}
                if last
                else None
            ),
        }
    return {"running": False}


@router.get("/watch/events")
async def watch_events(request: Request):
    """SSE stream of file watcher events with heartbeat and timeout."""
    max_duration = 300  # 5 minutes
    heartbeat_interval = 15  # seconds
    # M12 fix: poll_interval dropped from 1s → 0.1s so watcher events reach
    # the UI within ~100ms instead of ~1s. Heartbeat cadence is unchanged.
    poll_interval = 0.1

    async def generator():
        start = time.monotonic()
        last_heartbeat = start

        while True:
            now = time.monotonic()
            if now - start > max_duration:
                break
            if await request.is_disconnected():
                break

            watcher = request.app.state.watcher
            if watcher and watcher.is_running:
                events = watcher.drain_events()
                for ev in events:
                    payload = json.dumps(
                        {"path": ev.path, "type": ev.event_type, "timestamp": ev.timestamp}
                    )
                    yield f"data: {payload}\n\n"

            if now - last_heartbeat >= heartbeat_interval:
                yield ": keepalive\n\n"
                last_heartbeat = now

            await asyncio.sleep(poll_interval)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
