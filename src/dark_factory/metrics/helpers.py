"""Fire-and-forget helpers for instrumentation sites.

Every helper reads the module-level ``_metrics_recorder`` from
``dark_factory.agents.tools`` and calls it if present (Postgres write),
and also fires the matching Prometheus observer (always on).
Never raises. Use these at hot paths (swarm tool calls, memory operations,
artifact writes) so the instrumentation sites stay one-liners and
downstream metric failures can never take the pipeline down.
"""

from __future__ import annotations

from typing import Any


def _recorder():
    try:
        from dark_factory.agents import tools as _tools_mod

        return _tools_mod._metrics_recorder
    except Exception:
        return None


# ── Tool calls ──────────────────────────────────────────────────────────────


def record_tool_call(**fields: Any) -> None:
    try:
        from dark_factory.metrics.prometheus import observe_tool_call

        observe_tool_call(**fields)
    except Exception:
        pass

    rec = _recorder()
    if rec is None:
        return
    try:
        rec.record_tool_call(**fields)
    except Exception:
        pass


# ── Agent stats rollup ─────────────────────────────────────────────────────


def record_agent_stats(**fields: Any) -> None:
    # Prometheus breakdown: bump counters once per rollup row so we
    # continuously track aggregate agent usage by name.
    try:
        from dark_factory.metrics.prometheus import (
            agent_activations_total,
            _label,
        )

        agent = _label(fields.get("agent"))
        activations = int(fields.get("activations") or 0)
        if activations:
            agent_activations_total.labels(agent=agent).inc(activations)
    except Exception:
        pass

    rec = _recorder()
    if rec is None:
        return
    try:
        rec.record_agent_stats(**fields)
    except Exception:
        pass


# ── Spec decomposition ─────────────────────────────────────────────────────


def record_decomposition_stats(**fields: Any) -> None:
    try:
        from dark_factory.metrics.prometheus import observe_spec_plan

        fallback = bool(fields.get("fallback"))
        empty = bool(fields.get("empty_result"))
        truncated = bool(fields.get("truncated"))
        if fallback:
            outcome = "fallback"
        elif empty:
            outcome = "empty"
        elif truncated:
            outcome = "truncated"
        else:
            outcome = "success"
        observe_spec_plan(
            outcome=outcome,
            sub_spec_count=fields.get("planned_sub_specs_count"),
        )
    except Exception:
        pass

    rec = _recorder()
    if rec is None:
        return
    try:
        rec.record_decomposition_stats(**fields)
    except Exception:
        pass


# ── Memory ops ─────────────────────────────────────────────────────────────


def record_memory_operation(**fields: Any) -> None:
    try:
        from dark_factory.metrics.prometheus import observe_memory_op

        observe_memory_op(**fields)
    except Exception:
        pass

    rec = _recorder()
    if rec is None:
        return
    try:
        rec.record_memory_operation(**fields)
    except Exception:
        pass


def fetch_memory_effectiveness(*, days: int = 7) -> dict:
    """Aggregate boost/demote/decay/recall counts over the last N days.

    Reads from the Postgres ``memory_operations`` table populated by
    :func:`record_memory_operation`. Returns zeros when Postgres is
    disabled or no recorder is installed — consumers should treat
    this as a best-effort observability query, not a source of truth.
    """
    zero = {
        "window_days": days,
        "boosted": 0,
        "demoted": 0,
        "decays": 0,
        "total_recalls": 0,
        "boost_rate": 0.0,
    }

    rec = _recorder()
    if rec is None:
        return zero

    fetcher = getattr(rec, "fetch_memory_effectiveness", None)
    if fetcher is None:
        return zero
    try:
        return fetcher(days=days) or zero
    except Exception:
        return zero


# ── Incidents ──────────────────────────────────────────────────────────────


def record_incident(**fields: Any) -> None:
    try:
        from dark_factory.metrics.prometheus import observe_incident

        observe_incident(
            category=fields.get("category", "other"),
            severity=fields.get("severity", "error"),
        )
    except Exception:
        pass

    rec = _recorder()
    if rec is None:
        return
    try:
        rec.record_incident(**fields)
    except Exception:
        pass


# ── Artifact writes ─────────────────────────────────────────────────────────


def record_artifact_write(**fields: Any) -> None:
    try:
        from dark_factory.metrics.prometheus import observe_artifact_write

        observe_artifact_write(**fields)
    except Exception:
        pass

    rec = _recorder()
    if rec is None:
        return
    try:
        rec.record_artifact_write(**fields)
    except Exception:
        pass


# ── Swarm feature events ───────────────────────────────────────────────────


def record_swarm_feature_event(**fields: Any) -> None:
    try:
        from dark_factory.metrics.prometheus import observe_feature_event

        observe_feature_event(
            event=fields.get("event", ""),
            status=fields.get("status"),
            duration_seconds=fields.get("duration_seconds"),
        )
    except Exception:
        pass

    rec = _recorder()
    if rec is None:
        return
    try:
        rec.record_swarm_feature_event(**fields)
    except Exception:
        pass


# ── Background loop sample ─────────────────────────────────────────────────


def record_background_loop_sample(**fields: Any) -> None:
    try:
        from dark_factory.metrics.prometheus import observe_bg_loop_sample

        observe_bg_loop_sample(**fields)
    except Exception:
        pass

    rec = _recorder()
    if rec is None:
        return
    try:
        rec.record_background_loop_sample(**fields)
    except Exception:
        pass
