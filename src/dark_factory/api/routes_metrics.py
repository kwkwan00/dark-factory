"""REST endpoints for the Postgres metrics store.

All endpoints degrade gracefully when Postgres is disabled or the client
isn't initialised — they return a ``{"enabled": false, ...}`` payload with
HTTP 200 so the frontend can render an informative empty state instead of
bailing out on a 503.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Query, Request

log = structlog.get_logger()

router = APIRouter()


def _repo_or_none(request: Request):
    client = getattr(request.app.state, "metrics_client", None)
    if client is None:
        return None
    from dark_factory.metrics.repository import MetricsRepository

    return MetricsRepository(client)


def _disabled_payload(extra: dict | None = None) -> dict:
    payload = {"enabled": False, "reason": "postgres metrics store is disabled"}
    if extra:
        payload.update(extra)
    return payload


def _guarded(fn_name: str, func):
    """Shared try/except wrapper turning repo exceptions into HTTP 503."""
    try:
        return func()
    except Exception as exc:
        log.warning(f"{fn_name}_failed", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=f"Metrics query failed: {exc}",
        ) from exc


@router.get("/metrics/summary")
def metrics_summary(request: Request):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({
            "runs": {}, "llm": {}, "evals": {},
            "incidents": {}, "decomposition": {},
        })
    data = _guarded("metrics_summary", repo.query_summary)
    return {"enabled": True, **data}


@router.get("/metrics/runs")
def metrics_runs(
    request: Request,
    limit: int = Query(default=20, ge=1, le=200),
):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"runs": []})
    rows = _guarded("metrics_runs", lambda: repo.query_recent_runs(limit=limit))
    return {"enabled": True, "runs": rows}


@router.get("/metrics/eval_trend")
def metrics_eval_trend(
    request: Request,
    metric_name: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"events": []})
    events = _guarded(
        "metrics_eval_trend",
        lambda: repo.query_eval_trend(metric_name=metric_name, limit=limit),
    )
    return {"enabled": True, "metric_name": metric_name, "events": events}


@router.get("/metrics/llm_usage")
def metrics_llm_usage(
    request: Request,
    group_by: str = Query(default="model", pattern=r"^(model|phase|client)$"),
    limit: int = Query(default=50, ge=1, le=200),
):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"buckets": []})
    buckets = _guarded(
        "metrics_llm_usage",
        lambda: repo.query_llm_usage(group_by=group_by, limit=limit),
    )
    return {"enabled": True, "group_by": group_by, "buckets": buckets}


@router.get("/metrics/swarm_features")
def metrics_swarm_features(
    request: Request,
    # M1 fix: bound the run_id with the same regex the other routes
    # use. Defense-in-depth — the downstream repo uses parameterised
    # SQL so there's no injection, but the regex rejects malformed
    # values before the DB round-trip and caps the length to 128.
    run_id: str | None = Query(
        default=None, pattern=r"^[A-Za-z0-9_\-]{1,128}$"
    ),
    limit: int = Query(default=100, ge=1, le=500),
):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"events": []})
    events = _guarded(
        "metrics_swarm_features",
        lambda: repo.query_swarm_features(run_id=run_id, limit=limit),
    )
    return {"enabled": True, "run_id": run_id, "events": events}


# ── Extended endpoints ─────────────────────────────────────────────────────


@router.get("/metrics/cost_rollup")
def metrics_cost_rollup(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"per_run": [], "per_phase": [], "per_model": []})
    data = _guarded(
        "metrics_cost_rollup", lambda: repo.query_cost_rollup(limit=limit)
    )
    return {"enabled": True, **data}


@router.get("/metrics/throughput")
def metrics_throughput(
    request: Request,
    days: int = Query(default=30, ge=1, le=365),
):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"days": []})
    rows = _guarded("metrics_throughput", lambda: repo.query_throughput(days=days))
    return {"enabled": True, "days": rows}


@router.get("/metrics/quality")
def metrics_quality(request: Request):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"per_metric": []})
    data = _guarded("metrics_quality", repo.query_quality)
    return {"enabled": True, **data}


@router.get("/metrics/incidents")
def metrics_incidents(
    request: Request,
    category: str | None = Query(default=None),
    unresolved_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=500),
):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"incidents": []})
    rows = _guarded(
        "metrics_incidents",
        lambda: repo.query_incidents(
            limit=limit, category=category, unresolved_only=unresolved_only
        ),
    )
    return {"enabled": True, "incidents": rows}


@router.get("/metrics/agent_stats")
def metrics_agent_stats(
    request: Request,
    # M1 fix: bound the run_id with a regex pattern.
    run_id: str | None = Query(
        default=None, pattern=r"^[A-Za-z0-9_\-]{1,128}$"
    ),
    limit: int = Query(default=100, ge=1, le=500),
):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"agents": []})
    rows = _guarded(
        "metrics_agent_stats",
        lambda: repo.query_agent_stats(run_id=run_id, limit=limit),
    )
    return {"enabled": True, "agents": rows}


@router.get("/metrics/tool_calls")
def metrics_tool_calls(
    request: Request,
    group_by: str = Query(default="tool", pattern=r"^(tool|agent|feature)$"),
    limit: int = Query(default=50, ge=1, le=500),
):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"buckets": []})
    buckets = _guarded(
        "metrics_tool_calls",
        lambda: repo.query_tool_calls(group_by=group_by, limit=limit),
    )
    return {"enabled": True, "group_by": group_by, "buckets": buckets}


@router.get("/metrics/memory_activity")
def metrics_memory_activity(request: Request):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({
            "per_operation": [], "per_type": [], "summary": {}
        })
    data = _guarded("metrics_memory_activity", repo.query_memory_activity)
    return {"enabled": True, **data}


@router.get("/metrics/decomposition")
def metrics_decomposition(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"summary": {}, "rows": []})
    data = _guarded("metrics_decomposition", lambda: repo.query_decomposition(limit=limit))
    return {"enabled": True, **data}


@router.get("/metrics/artifacts")
def metrics_artifacts(request: Request):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({
            "summary": {}, "per_language": [], "per_feature": []
        })
    data = _guarded("metrics_artifacts", repo.query_artifacts)
    return {"enabled": True, **data}


@router.get("/metrics/runs/{run_id}")
def metrics_run_detail(
    request: Request,
    run_id: str,
):
    """Return aggregated per-run metrics for the run detail popup.

    All sub-queries are scoped to a single ``run_id`` (LLM totals,
    swarm events, agent stats, tool calls, incidents, eval metrics,
    artifacts, decomposition). Returns 404 if the run doesn't exist.
    """
    # Reject anything that isn't a safe run id up front so we never
    # hand arbitrary strings to SQL even through parameterized queries.
    if not _is_valid_run_id(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format")

    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"run_id": run_id})
    detail = _guarded(
        "metrics_run_detail", lambda: repo.query_run_detail(run_id)
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id!r} not found in metrics store",
        )
    return {"enabled": True, **detail}


def _is_valid_run_id(run_id: str) -> bool:
    """Accept the format emitted by orchestrator.py:
    ``run-YYYYMMDD-HHMMSS-xxxx`` plus any alphanumeric/dash/underscore
    variant for flexibility. Rejects slashes, dots, and whitespace so
    it's safe to use in path params and in filesystem joins later.
    """
    import re

    return bool(re.fullmatch(r"[A-Za-z0-9_\-]{1,128}", run_id))


@router.get("/metrics/memory")
def metrics_memory(request: Request):
    """Return procedural memory graph observability data.

    Tier A dashboard. Aggregates three queries against
    :class:`MemoryRepository`:

    - ``counts_by_type`` — one entry per memory label with total
      count, relevance statistics, and a 10-bucket relevance
      histogram (for the bar chart in the UI).
    - ``top_recalled`` — the 10 most-recalled memories globally,
      used to surface the "workhorses" of the graph.
    - ``recall_effectiveness`` — aggregate boost/demote/decay
      counts over the last 7 days, computed from the Postgres
      metrics store if available (zeros if disabled).

    Degrades gracefully when the memory store is disabled — returns
    ``{"enabled": false, ...}`` with HTTP 200 so the frontend can
    render an informative empty state instead of a 503.
    """
    memory_repo = getattr(request.app.state, "memory_repo", None)
    if memory_repo is None:
        return {
            "enabled": False,
            "reason": "memory store is disabled or not initialised",
            "counts_by_type": {},
            "top_recalled": [],
            "recall_effectiveness": {
                "window_days": 7,
                "boosted": 0,
                "demoted": 0,
                "decays": 0,
                "total_recalls": 0,
                "boost_rate": 0.0,
            },
        }

    try:
        counts_by_type = memory_repo.get_memory_stats()
    except Exception as exc:
        log.warning("memory_stats_query_failed", error=str(exc))
        raise HTTPException(
            status_code=503, detail=f"Memory stats query failed: {exc}"
        ) from exc

    try:
        top_recalled = memory_repo.get_top_recalled_memories(limit=10)
    except Exception as exc:
        log.warning("memory_top_recalled_query_failed", error=str(exc))
        top_recalled = []

    try:
        effectiveness = memory_repo.get_recall_effectiveness(days=7)
    except Exception as exc:
        log.warning("memory_effectiveness_query_failed", error=str(exc))
        effectiveness = {
            "window_days": 7,
            "boosted": 0,
            "demoted": 0,
            "decays": 0,
            "total_recalls": 0,
            "boost_rate": 0.0,
        }

    return {
        "enabled": True,
        "counts_by_type": counts_by_type,
        "top_recalled": top_recalled,
        "recall_effectiveness": effectiveness,
    }


@router.get("/metrics/episodes/{run_id}")
def metrics_episodes(
    request: Request,
    run_id: str,
    feature: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Return episodic memory records for a run.

    Episodes are autobiographical per-feature records produced by
    the orchestrator after each swarm finishes. Each record carries
    a narrative summary, key events, the agents that ran, eval
    scores, and timing info. The Run Detail popup's Episodes tab
    renders these into a timeline.

    Results are sorted newest-first and can be filtered to a single
    feature via the ``feature`` query param. JSON fields stored as
    strings in Neo4j (``key_events_json``, ``tool_calls_json``,
    ``eval_scores_json``) are decoded here so the frontend sees
    structured objects directly.
    """
    if not _is_valid_run_id(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format")

    memory_repo = getattr(request.app.state, "memory_repo", None)
    if memory_repo is None:
        return {
            "enabled": False,
            "run_id": run_id,
            "episodes": [],
            "reason": "memory store is disabled or not initialised",
        }

    try:
        raw = memory_repo.get_episodes_for_run(
            run_id=run_id,
            feature=feature,
            limit=limit,
        )
    except Exception as exc:
        log.warning("metrics_episodes_query_failed", run_id=run_id, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=f"Episode query failed: {exc}",
        ) from exc

    import json as _json

    decoded: list[dict] = []
    for ep in raw:
        item: dict = {
            "id": ep.get("id", ""),
            "run_id": ep.get("run_id", run_id),
            "feature": ep.get("feature", ""),
            "outcome": ep.get("outcome", ""),
            "summary": ep.get("summary", ""),
            "turns_used": ep.get("turns_used", 0),
            "duration_seconds": ep.get("duration_seconds", 0.0),
            "spec_ids": ep.get("spec_ids", []) or [],
            "agents_visited": ep.get("agents_visited", []) or [],
            "started_at": ep.get("started_at", ""),
            "ended_at": ep.get("ended_at", ""),
            "key_events": [],
            "tool_calls_summary": {},
            "final_eval_scores": {},
        }
        for raw_key, out_key in (
            ("key_events_json", "key_events"),
            ("tool_calls_json", "tool_calls_summary"),
            ("eval_scores_json", "final_eval_scores"),
        ):
            raw_val = ep.get(raw_key)
            if isinstance(raw_val, str) and raw_val.strip():
                try:
                    item[out_key] = _json.loads(raw_val)
                except Exception:
                    pass
        decoded.append(item)

    return {
        "enabled": True,
        "run_id": run_id,
        "feature": feature,
        "episodes": decoded,
    }


@router.get("/metrics/background_loop")
def metrics_background_loop(
    request: Request,
    limit: int = Query(default=200, ge=1, le=2000),
):
    repo = _repo_or_none(request)
    if repo is None:
        return _disabled_payload({"samples": []})
    rows = _guarded(
        "metrics_background_loop",
        lambda: repo.query_background_loop(limit=limit),
    )
    return {"enabled": True, "samples": rows}
