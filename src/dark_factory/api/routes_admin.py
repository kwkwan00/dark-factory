"""Admin endpoints for destructive operations.

These endpoints wipe state from the docker-compose managed stores
(Neo4j, Qdrant, Postgres) plus the filesystem output directory. They
are guarded by an explicit ``?confirm=yes`` query parameter and refuse
to run while a pipeline is in progress.

Intended for local development and CI fixtures — never expose to the
open internet without an auth layer in front.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException, Query, Request

log = structlog.get_logger()

router = APIRouter()


# The tables truncated by ``POST /admin/clear-all``. Listed leaves-first
# so CASCADE doesn't have to do extra work; CASCADE is still used so any
# future foreign keys are handled automatically.
_METRICS_TABLES = [
    "background_loop_samples",
    "artifact_writes",
    "incidents",
    "memory_operations",
    "decomposition_stats",
    "agent_stats",
    "tool_calls",
    "swarm_feature_events",
    "llm_calls",
    "eval_metrics",
    "progress_events",
    "pipeline_runs",
]


def _clear_neo4j(request: Request) -> dict:
    """Wipe every node + relationship from the shared Neo4j database.

    This clears both the knowledge graph (Specs, Requirements, IMPLEMENTS,
    DEPENDS_ON, …) and procedural memory (Pattern, Mistake, Solution,
    Strategy, Run, EvalResult) since Community Edition holds them in the
    same DB. Returns a count of deleted nodes.
    """
    neo4j_client = request.app.state.neo4j_client
    with neo4j_client.session() as session:
        # Count before deletion so we have something honest to return
        # (DETACH DELETE doesn't emit a row count via RETURN).
        result = session.run("MATCH (n) RETURN count(n) AS cnt")
        nodes_before = int(result.single()["cnt"] or 0)
        session.run("MATCH (n) DETACH DELETE n")
    return {"nodes_deleted": nodes_before}


def _clear_qdrant(request: Request) -> dict:
    """Delete every dark-factory Qdrant collection, then recreate them
    empty so the next pipeline run can upsert without errors.

    Uses the ``COLLECTION_INDEXES`` keys from the collections module as
    the source of truth for which suffixes belong to this project,
    avoiding any chance of deleting collections owned by another app
    sharing the Qdrant instance.
    """
    vector_repo = getattr(request.app.state, "vector_repo", None)
    if vector_repo is None:
        return {"status": "disabled"}

    from dark_factory.vector.collections import (
        COLLECTION_INDEXES,
        ensure_collections,
    )

    wrapper = vector_repo._client  # internal access — this is admin-only
    client = wrapper.client
    cleared: list[str] = []
    skipped: list[dict] = []
    for suffix in COLLECTION_INDEXES.keys():
        name = wrapper.collection_name(suffix)
        try:
            if client.collection_exists(name):
                client.delete_collection(name)
                cleared.append(name)
        except Exception as exc:
            skipped.append({"collection": name, "error": str(exc)})
            log.warning(
                "admin_qdrant_delete_collection_failed",
                collection=name,
                error=str(exc),
            )

    # Recreate the collection shells so subsequent runs don't need a
    # restart to re-init them.
    try:
        ensure_collections(wrapper)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("admin_qdrant_recreate_failed", error=str(exc))
        skipped.append({"collection": "(recreate)", "error": str(exc)})

    out: dict = {"collections_cleared": cleared}
    if skipped:
        out["skipped"] = skipped
    return out


def _clear_postgres(request: Request) -> dict:
    """TRUNCATE every metrics table. Uses ``RESTART IDENTITY CASCADE`` so
    BIGSERIAL ids reset to 1 and any future foreign keys are honoured.
    """
    metrics_client = getattr(request.app.state, "metrics_client", None)
    if metrics_client is None:
        return {"status": "disabled"}

    truncated: list[str] = []
    skipped: list[dict] = []
    with metrics_client.connection() as conn:
        with conn.cursor() as cur:
            for table in _METRICS_TABLES:
                try:
                    # table names are a hardcoded module-level list, not
                    # user input, so the f-string is safe here. psycopg
                    # does not allow binding identifiers via placeholders.
                    cur.execute(
                        f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"
                    )
                    truncated.append(table)
                except Exception as exc:
                    skipped.append({"table": table, "error": str(exc)})
                    log.warning(
                        "admin_postgres_truncate_failed",
                        table=table,
                        error=str(exc),
                    )
        conn.commit()

    out: dict = {"tables_truncated": truncated}
    if skipped:
        out["skipped"] = skipped
    return out


def _clear_prometheus(request: Request) -> dict:
    """Wipe every dark-factory time series from the remote Prometheus
    TSDB, then reset the in-process ``prometheus_client`` collectors so
    the next scrape doesn't immediately replay the old numbers.

    Requires the Prometheus server to be started with
    ``--web.enable-admin-api`` (docker-compose sets this). When the
    server is disabled in settings or unreachable, the in-process
    collectors are still reset and the TSDB step is reported as skipped.

    The match query ``{__name__=~"dark_factory_.+"}`` deliberately
    scopes deletion to THIS app's collectors — we never touch
    ``python_*`` / ``process_*`` / ``up`` / other jobs' series.
    """
    # Step 1: always reset the in-process collectors, even if the
    # remote delete is disabled/unreachable. This is the cheap part.
    from dark_factory.metrics import prometheus as prom_module

    try:
        reset = prom_module.reset_all()
    except Exception as exc:
        log.warning("admin_prometheus_reset_inprocess_failed", error=str(exc))
        reset = {"error": str(exc)}

    settings = request.app.state.settings
    prom_cfg = getattr(settings, "prometheus", None)
    if prom_cfg is None or not getattr(prom_cfg, "enabled", False):
        return {
            "status": "in_process_only",
            "reason": "prometheus.enabled=false",
            "in_process": reset,
        }

    # Step 2: remote TSDB delete via the admin API.
    import httpx

    base = prom_cfg.url.rstrip("/")
    match = '{__name__=~"dark_factory_.+"}'
    delete_url = f"{base}/api/v1/admin/tsdb/delete_series"
    clean_url = f"{base}/api/v1/admin/tsdb/clean_tombstones"

    result: dict = {"in_process": reset, "url": base}
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(delete_url, params={"match[]": match})
            if resp.status_code == 204:
                result["series_deleted"] = True
            elif resp.status_code == 404:
                # admin API not enabled on the server — report clearly
                # rather than hiding as a generic failure.
                result["status"] = "admin_api_disabled"
                result["hint"] = (
                    "Prometheus server must be started with "
                    "--web.enable-admin-api for delete_series to work."
                )
                return result
            else:
                result["status"] = "delete_failed"
                result["http_status"] = resp.status_code
                result["body"] = resp.text[:500]
                return result

            # Compact the tombstones we just wrote so disk space is
            # actually reclaimed. Also 204 on success.
            resp = client.post(clean_url)
            result["tombstones_cleaned"] = resp.status_code == 204
    except httpx.HTTPError as exc:
        result["status"] = "unreachable"
        result["error"] = str(exc)
        log.warning("admin_prometheus_unreachable", url=base, error=str(exc))
        return result

    result["status"] = "completed"
    return result


def _clear_output_dir(request: Request) -> dict:
    """Remove everything under the configured pipeline output directory.

    Safety: refuses to delete anything outside the project working
    directory, even if an operator has pointed ``output_dir`` at ``/``
    or similar. The directory is recreated empty after deletion so a
    subsequent run can write to it immediately.
    """
    settings = request.app.state.settings
    output_dir = Path(settings.pipeline.output_dir).resolve()
    cwd = Path.cwd().resolve()

    # Hard safety: the output dir MUST live inside the working directory.
    # This prevents a misconfigured config.toml from wiping unrelated
    # files on the host.
    try:
        is_safe = output_dir.is_relative_to(cwd)
    except Exception:
        is_safe = False
    if not is_safe:
        raise ValueError(
            f"Refusing to clear output dir {output_dir}: not inside cwd {cwd}"
        )

    files_deleted = 0
    bytes_freed = 0
    if output_dir.exists():
        for item in output_dir.rglob("*"):
            if item.is_file():
                try:
                    files_deleted += 1
                    bytes_freed += item.stat().st_size
                except OSError:
                    pass
        shutil.rmtree(output_dir, ignore_errors=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "files_deleted": files_deleted,
        "bytes_freed": bytes_freed,
        "path": str(output_dir),
    }


@router.post("/admin/clear-all")
async def admin_clear_all(
    request: Request,
    confirm: str = Query(
        default="",
        description="Must be set to 'yes' to actually run the wipe.",
    ),
    include_output_dir: bool = Query(
        default=True,
        description="Whether to wipe the pipeline output directory too.",
    ),
    include_prometheus: bool = Query(
        default=True,
        description=(
            "Whether to wipe the Prometheus TSDB + reset in-process "
            "collectors. In-process collectors are always reset when this "
            "flag is true, even if the remote server is unreachable."
        ),
    ),
):
    """Nuclear reset: wipe every docker-compose managed store.

    Clears in this order:

    1. **Neo4j** — all nodes + relationships (knowledge graph AND
       procedural memory, since Community Edition uses one DB).
    2. **Qdrant** — every dark-factory collection, then recreated empty.
    3. **Postgres metrics store** — every metrics table TRUNCATEd with
       ``RESTART IDENTITY CASCADE``.
    4. **Prometheus** — delete every ``dark_factory_*`` time series from
       the remote TSDB and reset in-process collectors (optional).
    5. **Output directory** — deleted and recreated empty (optional).

    Safety rails:
    - Requires ``?confirm=yes`` or returns 400.
    - Returns 409 if a pipeline run is currently in progress — the
      caller must cancel the run before clearing state.
    - Continues through per-store errors and reports them in the
      response payload rather than short-circuiting on the first
      failure, so a partial wipe leaves actionable debug info.
    """
    if confirm != "yes":
        raise HTTPException(
            status_code=400,
            detail="Refusing to clear all data without ?confirm=yes",
        )

    run_lock: asyncio.Lock = request.app.state.run_lock
    if run_lock.locked():
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot clear data while a pipeline run is in progress. "
                "Cancel the run first via POST /api/agent/cancel."
            ),
        )

    remote = str(request.client.host) if request.client else "?"
    log.warning("admin_clear_all_starting", remote=remote)

    report: dict = {}
    errors: dict = {}

    # 1. Neo4j
    try:
        report["neo4j"] = await asyncio.to_thread(_clear_neo4j, request)
    except Exception as exc:
        errors["neo4j"] = str(exc)
        log.error("admin_clear_neo4j_failed", error=str(exc))

    # 2. Qdrant
    try:
        report["qdrant"] = await asyncio.to_thread(_clear_qdrant, request)
    except Exception as exc:
        errors["qdrant"] = str(exc)
        log.error("admin_clear_qdrant_failed", error=str(exc))

    # 3. Postgres metrics store
    try:
        report["postgres"] = await asyncio.to_thread(_clear_postgres, request)
    except Exception as exc:
        errors["postgres"] = str(exc)
        log.error("admin_clear_postgres_failed", error=str(exc))

    # 4. Prometheus (optional — defaults to yes). Runs before the
    # output dir so a failure here doesn't leave an orphaned wipe of
    # generated files while Prometheus still shows the old numbers.
    if include_prometheus:
        try:
            report["prometheus"] = await asyncio.to_thread(
                _clear_prometheus, request
            )
        except Exception as exc:
            errors["prometheus"] = str(exc)
            log.error("admin_clear_prometheus_failed", error=str(exc))
    else:
        report["prometheus"] = {"status": "skipped"}

    # 5. Output directory (optional — defaults to yes)
    if include_output_dir:
        try:
            report["output_dir"] = await asyncio.to_thread(
                _clear_output_dir, request
            )
        except Exception as exc:
            errors["output_dir"] = str(exc)
            log.error("admin_clear_output_failed", error=str(exc))
    else:
        report["output_dir"] = {"status": "skipped"}

    # Also clear the progress broker history + Prometheus gauges that
    # describe in-memory state, so the dashboard reflects the wipe
    # immediately without needing an app restart.
    try:
        from dark_factory.agents import tools as _tools_mod

        if _tools_mod._progress_broker is not None:
            _tools_mod._progress_broker.clear_history()
        report["progress_broker_history"] = {"cleared": True}
    except Exception as exc:  # pragma: no cover — defensive
        errors["progress_broker"] = str(exc)

    log.warning(
        "admin_clear_all_completed",
        remote=remote,
        ok=not errors,
        report=report,
        errors=errors,
    )

    return {
        "status": "completed" if not errors else "partial",
        "cleared": report,
        "errors": errors,
    }
