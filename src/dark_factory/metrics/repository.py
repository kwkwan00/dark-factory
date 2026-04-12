"""Typed read/write API on top of the Postgres metrics schema.

All methods are synchronous and thread-safe (connection pooling handles
concurrency). Write methods are intended to be called from the background
:class:`MetricsRecorder` thread and never from request handlers directly.
Read methods are safe to call from async FastAPI handlers — individual
queries return in <100ms on the expected data volume.
"""

from __future__ import annotations

from typing import Any

import structlog
from psycopg.types.json import Json

log = structlog.get_logger()


# Events that carry per-metric rubric data we want in ``eval_metrics``.
_EVAL_RUBRIC_EVENT = "eval_rubric"

# Events that carry feature lifecycle data we want in ``swarm_feature_events``.
_SWARM_FEATURE_EVENTS = {
    "feature_started": "started",
    "feature_completed": "completed",
    "feature_skipped": "skipped",
}


class MetricsRepository:
    """Read/write API over the metrics schema."""

    def __init__(self, client) -> None:
        self.client = client

    # ── Writes: pipeline runs ─────────────────────────────────────────────

    def record_pipeline_run_start(
        self,
        *,
        run_id: str,
        spec_count: int = 0,
        feature_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pipeline_runs (run_id, spec_count, feature_count, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        spec_count = EXCLUDED.spec_count,
                        feature_count = EXCLUDED.feature_count,
                        metadata = EXCLUDED.metadata
                    """,
                    (run_id, spec_count, feature_count, Json(metadata or {})),
                )
            conn.commit()

    def record_pipeline_run_end(
        self,
        *,
        run_id: str,
        status: str,
        pass_rate: float | None = None,
        duration_seconds: float | None = None,
        error: str | None = None,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_runs
                       SET ended_at = NOW(),
                           status = %s,
                           pass_rate = %s,
                           duration_seconds = %s,
                           error = %s
                     WHERE run_id = %s
                    """,
                    (status, pass_rate, duration_seconds, error, run_id),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        """
                        INSERT INTO pipeline_runs
                            (run_id, status, pass_rate, duration_seconds, error, ended_at)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        """,
                        (run_id, status, pass_rate, duration_seconds, error),
                    )
            conn.commit()

    # ── Writes: raw progress audit log ────────────────────────────────────

    def record_progress_event(
        self,
        *,
        event: str,
        run_id: str | None = None,
        feature: str | None = None,
        agent: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO progress_events (run_id, event, feature, agent, payload)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (run_id, event, feature, agent, Json(payload or {})),
                )
            conn.commit()

    # ── Writes: eval rubric metrics ───────────────────────────────────────

    def record_eval_metric(
        self,
        *,
        metric_name: str,
        score: float,
        passed: bool,
        eval_type: str = "spec",
        run_id: str | None = None,
        requirement_id: str | None = None,
        spec_id: str | None = None,
        threshold: float | None = None,
        attempt: int | None = None,
        role: str | None = None,
        reason: str | None = None,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO eval_metrics (
                        run_id, requirement_id, spec_id, eval_type,
                        metric_name, score, passed, threshold,
                        attempt, role, reason
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id, requirement_id, spec_id, eval_type,
                        metric_name, score, passed, threshold,
                        attempt, role, reason,
                    ),
                )
            conn.commit()

    # ── Writes: LLM calls ─────────────────────────────────────────────────

    def record_llm_call(
        self,
        *,
        client: str,
        model: str,
        phase: str | None = None,
        run_id: str | None = None,
        prompt_chars: int | None = None,
        completion_chars: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_input_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
        system_prompt_chars: int | None = None,
        max_tokens_requested: int | None = None,
        temperature: float | None = None,
        latency_seconds: float | None = None,
        time_to_first_token_seconds: float | None = None,
        queue_wait_seconds: float | None = None,
        retry_count: int = 0,
        stop_reason: str | None = None,
        http_status: int | None = None,
        rate_limited: bool = False,
        cost_usd: float | None = None,
        error: str | None = None,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_calls (
                        run_id, client, model, phase,
                        prompt_chars, completion_chars,
                        input_tokens, output_tokens,
                        cache_read_input_tokens, cache_creation_input_tokens,
                        system_prompt_chars, max_tokens_requested, temperature,
                        latency_seconds, time_to_first_token_seconds, queue_wait_seconds,
                        retry_count, stop_reason, http_status, rate_limited,
                        cost_usd, error
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id, client, model, phase,
                        prompt_chars, completion_chars,
                        input_tokens, output_tokens,
                        cache_read_input_tokens, cache_creation_input_tokens,
                        system_prompt_chars, max_tokens_requested, temperature,
                        latency_seconds, time_to_first_token_seconds, queue_wait_seconds,
                        retry_count, stop_reason, http_status, rate_limited,
                        cost_usd, error,
                    ),
                )
            conn.commit()

    # ── Writes: swarm feature lifecycle ───────────────────────────────────

    def record_swarm_feature_event(
        self,
        *,
        feature: str,
        event: str,
        run_id: str | None = None,
        status: str | None = None,
        artifact_count: int | None = None,
        test_count: int | None = None,
        handoff_count: int | None = None,
        layer: int | None = None,
        error: str | None = None,
        duration_seconds: float | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        agent_transitions: int | None = None,
        unique_agents_visited: int | None = None,
        planner_calls: int | None = None,
        coder_calls: int | None = None,
        reviewer_calls: int | None = None,
        tester_calls: int | None = None,
        tool_call_count: int | None = None,
        tool_failure_count: int | None = None,
        deep_agent_invocations: int | None = None,
        deep_agent_timeout_count: int | None = None,
        subprocess_spawn_count: int | None = None,
        worker_crash_count: int | None = None,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO swarm_feature_events (
                        run_id, feature, event, status,
                        artifact_count, test_count, handoff_count,
                        layer, error, duration_seconds,
                        started_at, ended_at,
                        agent_transitions, unique_agents_visited,
                        planner_calls, coder_calls, reviewer_calls, tester_calls,
                        tool_call_count, tool_failure_count,
                        deep_agent_invocations, deep_agent_timeout_count,
                        subprocess_spawn_count, worker_crash_count
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id, feature, event, status,
                        artifact_count, test_count, handoff_count,
                        layer, error, duration_seconds,
                        started_at, ended_at,
                        agent_transitions, unique_agents_visited,
                        planner_calls, coder_calls, reviewer_calls, tester_calls,
                        tool_call_count, tool_failure_count,
                        deep_agent_invocations, deep_agent_timeout_count,
                        subprocess_spawn_count, worker_crash_count,
                    ),
                )
            conn.commit()

    # ── Writes: tool calls ────────────────────────────────────────────────

    def record_tool_call(
        self,
        *,
        tool: str,
        run_id: str | None = None,
        feature: str | None = None,
        agent: str | None = None,
        success: bool | None = None,
        latency_seconds: float | None = None,
        args_chars: int | None = None,
        result_chars: int | None = None,
        error: str | None = None,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tool_calls (
                        run_id, feature, agent, tool, success,
                        latency_seconds, args_chars, result_chars, error
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id, feature, agent, tool, success,
                        latency_seconds, args_chars, result_chars, error,
                    ),
                )
            conn.commit()

    # ── Writes: agent stats rollup ────────────────────────────────────────

    def record_agent_stats(
        self,
        *,
        agent: str,
        run_id: str | None = None,
        feature: str | None = None,
        activations: int = 0,
        tool_calls: int = 0,
        decisions: int = 0,
        handoffs_in: int = 0,
        handoffs_out: int = 0,
        total_time_seconds: float | None = None,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_stats (
                        run_id, feature, agent,
                        activations, tool_calls, decisions,
                        handoffs_in, handoffs_out, total_time_seconds
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id, feature, agent,
                        activations, tool_calls, decisions,
                        handoffs_in, handoffs_out, total_time_seconds,
                    ),
                )
            conn.commit()

    # ── Writes: decomposition stats ───────────────────────────────────────

    def record_decomposition_stats(
        self,
        *,
        run_id: str | None = None,
        requirement_id: str | None = None,
        requirement_title: str | None = None,
        planned_sub_specs_count: int = 0,
        fallback: bool = False,
        empty_result: bool = False,
        truncated: bool = False,
        depends_on_declared: int = 0,
        depends_on_resolved: int = 0,
        depends_on_unresolved: int = 0,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO decomposition_stats (
                        run_id, requirement_id, requirement_title,
                        planned_sub_specs_count, fallback, empty_result, truncated,
                        depends_on_declared, depends_on_resolved, depends_on_unresolved
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id, requirement_id, requirement_title,
                        planned_sub_specs_count, fallback, empty_result, truncated,
                        depends_on_declared, depends_on_resolved, depends_on_unresolved,
                    ),
                )
            conn.commit()

    # ── Writes: memory operations ─────────────────────────────────────────

    def record_memory_operation(
        self,
        *,
        operation: str,
        run_id: str | None = None,
        memory_type: str | None = None,
        memory_id: str | None = None,
        source_feature: str | None = None,
        count: int | None = None,
        delta: float | None = None,
        latency_seconds: float | None = None,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_operations (
                        run_id, operation, memory_type, memory_id,
                        source_feature, count, delta, latency_seconds
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id, operation, memory_type, memory_id,
                        source_feature, count, delta, latency_seconds,
                    ),
                )
            conn.commit()

    # ── Writes: incidents ─────────────────────────────────────────────────

    def record_incident(
        self,
        *,
        category: str,
        severity: str,
        message: str,
        run_id: str | None = None,
        stack: str | None = None,
        phase: str | None = None,
        feature: str | None = None,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO incidents (
                        run_id, category, severity, message, stack, phase, feature
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (run_id, category, severity, message, stack, phase, feature),
                )
            conn.commit()

    # ── Writes: artifact writes ───────────────────────────────────────────

    def record_artifact_write(
        self,
        *,
        file_path: str,
        run_id: str | None = None,
        feature: str | None = None,
        spec_id: str | None = None,
        language: str | None = None,
        bytes_written: int = 0,
        is_test: bool = False,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO artifact_writes (
                        run_id, feature, spec_id, file_path,
                        language, bytes_written, is_test
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id, feature, spec_id, file_path,
                        language, bytes_written, is_test,
                    ),
                )
            conn.commit()

    # ── Writes: background loop samples ───────────────────────────────────

    def record_background_loop_sample(
        self,
        *,
        active_task_count: int = 0,
        pending_task_count: int = 0,
        completed_task_count: int = 0,
        loop_restarts: int = 0,
    ) -> None:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO background_loop_samples (
                        active_task_count, pending_task_count,
                        completed_task_count, loop_restarts
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        active_task_count, pending_task_count,
                        completed_task_count, loop_restarts,
                    ),
                )
            conn.commit()

    # ── Single entry point for the recorder's progress-event dispatch ────

    def ingest_progress_event(
        self,
        *,
        event_dict: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        event_name = str(event_dict.get("event", ""))
        if not event_name:
            return

        feature = event_dict.get("feature")
        agent = event_dict.get("agent")
        payload = {
            k: v for k, v in event_dict.items() if k not in {"event"}
        }

        self.record_progress_event(
            event=event_name,
            run_id=run_id,
            feature=feature if isinstance(feature, str) else None,
            agent=agent if isinstance(agent, str) else None,
            payload=payload,
        )

        if event_name == _EVAL_RUBRIC_EVENT:
            self._ingest_eval_rubric(event_dict, run_id=run_id)

        if event_name in _SWARM_FEATURE_EVENTS:
            self._ingest_swarm_feature(event_dict, event_name, run_id=run_id)

    def _ingest_eval_rubric(
        self, event_dict: dict[str, Any], *, run_id: str | None
    ) -> None:
        metrics = event_dict.get("metrics") or []
        if not isinstance(metrics, list):
            return
        threshold = _as_float(event_dict.get("threshold"))
        attempt = _as_int(event_dict.get("attempt"))
        role = event_dict.get("role") if isinstance(event_dict.get("role"), str) else None
        requirement_id = (
            event_dict.get("requirement_id")
            if isinstance(event_dict.get("requirement_id"), str)
            else None
        )
        # The spec stage emits the field as ``target_spec_id`` in its
        # ``eval_rubric`` progress event (see ``stages/spec.py``).
        # Fall back to ``spec_id`` for any callers that use the
        # canonical name directly.
        raw_spec_id = event_dict.get("spec_id") or event_dict.get("target_spec_id")
        spec_id = raw_spec_id if isinstance(raw_spec_id, str) else None
        eval_type = (
            event_dict.get("eval_type")
            if isinstance(event_dict.get("eval_type"), str)
            else "spec"
        )
        for raw in metrics:
            if not isinstance(raw, dict):
                continue
            name = raw.get("name")
            score = _as_float(raw.get("score"))
            if not isinstance(name, str) or score is None:
                continue
            self.record_eval_metric(
                metric_name=name,
                score=score,
                passed=bool(raw.get("passed", False)),
                eval_type=eval_type,
                run_id=run_id,
                requirement_id=requirement_id,
                spec_id=spec_id,
                threshold=threshold,
                attempt=attempt,
                role=role,
                reason=(raw.get("reason") or None),
            )

    def _ingest_swarm_feature(
        self,
        event_dict: dict[str, Any],
        event_name: str,
        *,
        run_id: str | None,
    ) -> None:
        feature = event_dict.get("feature")
        if not isinstance(feature, str):
            return
        normalised_event = _SWARM_FEATURE_EVENTS[event_name]
        self.record_swarm_feature_event(
            feature=feature,
            event=normalised_event,
            run_id=run_id,
            status=event_dict.get("status") if isinstance(event_dict.get("status"), str) else None,
            artifact_count=_as_int(event_dict.get("artifacts")),
            test_count=_as_int(event_dict.get("tests")),
            handoff_count=_as_int(event_dict.get("messages")),
            layer=_as_int(event_dict.get("layer")),
            error=event_dict.get("error") if isinstance(event_dict.get("error"), str) else None,
            duration_seconds=_as_float(event_dict.get("duration_seconds")),
            agent_transitions=_as_int(event_dict.get("agent_transitions")),
            unique_agents_visited=_as_int(event_dict.get("unique_agents_visited")),
            planner_calls=_as_int(event_dict.get("planner_calls")),
            coder_calls=_as_int(event_dict.get("coder_calls")),
            reviewer_calls=_as_int(event_dict.get("reviewer_calls")),
            tester_calls=_as_int(event_dict.get("tester_calls")),
            tool_call_count=_as_int(event_dict.get("tool_call_count")),
            tool_failure_count=_as_int(event_dict.get("tool_failure_count")),
            deep_agent_invocations=_as_int(event_dict.get("deep_agent_invocations")),
            deep_agent_timeout_count=_as_int(event_dict.get("deep_agent_timeout_count")),
            subprocess_spawn_count=_as_int(event_dict.get("subprocess_spawn_count")),
            worker_crash_count=_as_int(event_dict.get("worker_crash_count")),
        )

    # ── Reads ─────────────────────────────────────────────────────────────

    def query_summary(self) -> dict[str, Any]:
        """One-shot aggregate row for the Metrics tab header."""
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total_runs,
                        COUNT(*) FILTER (WHERE status = 'success') AS success_runs,
                        COUNT(*) FILTER (WHERE status = 'partial') AS partial_runs,
                        COUNT(*) FILTER (WHERE status = 'error') AS error_runs,
                        COUNT(*) FILTER (WHERE status = 'running') AS running_runs,
                        AVG(pass_rate) FILTER (WHERE pass_rate IS NOT NULL) AS avg_pass_rate,
                        AVG(duration_seconds) FILTER (WHERE duration_seconds IS NOT NULL) AS avg_duration_seconds
                    FROM pipeline_runs
                    """
                )
                runs_row = cur.fetchone() or {}

                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total_calls,
                        COALESCE(SUM(input_tokens), 0) AS input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS output_tokens,
                        COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_tokens,
                        COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
                        AVG(latency_seconds) FILTER (WHERE latency_seconds IS NOT NULL) AS avg_latency_seconds,
                        COUNT(*) FILTER (WHERE rate_limited) AS rate_limited_count,
                        COUNT(*) FILTER (WHERE error IS NOT NULL) AS error_count
                    FROM llm_calls
                    """
                )
                llm_row = cur.fetchone() or {}

                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total_evals,
                        AVG(score) AS avg_score,
                        COUNT(*) FILTER (WHERE passed) AS passed
                    FROM eval_metrics
                    """
                )
                eval_row = cur.fetchone() or {}

                cur.execute(
                    """
                    SELECT COUNT(*) AS open_incidents
                    FROM incidents
                    WHERE NOT resolved
                    """
                )
                incidents_row = cur.fetchone() or {}

                cur.execute(
                    """
                    SELECT
                        SUM(planned_sub_specs_count) AS total_sub_specs,
                        COUNT(*) AS requirements_planned,
                        COUNT(*) FILTER (WHERE fallback) AS planner_fallbacks
                    FROM decomposition_stats
                    """
                )
                decomp_row = cur.fetchone() or {}

        return {
            "runs": runs_row,
            "llm": llm_row,
            "evals": eval_row,
            "incidents": incidents_row,
            "decomposition": decomp_row,
        }

    def query_recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id, started_at, ended_at, status,
                           spec_count, feature_count, pass_rate,
                           duration_seconds, error
                    FROM pipeline_runs
                    ORDER BY started_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [_stringify_timestamps(r) for r in rows]

    def query_eval_trend(
        self, *, metric_name: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 1000))
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                if metric_name:
                    cur.execute(
                        """
                        SELECT timestamp, metric_name, score, passed, attempt,
                               run_id, spec_id
                        FROM eval_metrics
                        WHERE metric_name = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                        """,
                        (metric_name, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT timestamp, metric_name, score, passed, attempt,
                               run_id, spec_id
                        FROM eval_metrics
                        ORDER BY timestamp DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                rows = cur.fetchall()
        return [_stringify_timestamps(r) for r in rows]

    def query_llm_usage(
        self, *, group_by: str = "model", limit: int = 50
    ) -> list[dict[str, Any]]:
        if group_by not in {"model", "phase", "client"}:
            group_by = "model"
        limit = max(1, min(limit, 200))
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {group_by} AS bucket,
                           COUNT(*) AS calls,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens,
                           COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
                           AVG(latency_seconds) FILTER (WHERE latency_seconds IS NOT NULL) AS avg_latency_seconds
                    FROM llm_calls
                    GROUP BY {group_by}
                    ORDER BY calls DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def query_swarm_features(
        self, *, run_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                if run_id:
                    cur.execute(
                        """
                        SELECT run_id, feature, event, status,
                               artifact_count, test_count, handoff_count,
                               layer, error, duration_seconds,
                               agent_transitions, unique_agents_visited,
                               planner_calls, coder_calls, reviewer_calls, tester_calls,
                               tool_call_count, tool_failure_count,
                               deep_agent_invocations, deep_agent_timeout_count,
                               worker_crash_count, timestamp
                        FROM swarm_feature_events
                        WHERE run_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                        """,
                        (run_id, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT run_id, feature, event, status,
                               artifact_count, test_count, handoff_count,
                               layer, error, duration_seconds,
                               agent_transitions, unique_agents_visited,
                               planner_calls, coder_calls, reviewer_calls, tester_calls,
                               tool_call_count, tool_failure_count,
                               deep_agent_invocations, deep_agent_timeout_count,
                               worker_crash_count, timestamp
                        FROM swarm_feature_events
                        ORDER BY timestamp DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                rows = cur.fetchall()
        return [_stringify_timestamps(r) for r in rows]

    def query_cost_rollup(self, *, limit: int = 50) -> dict[str, Any]:
        """Per-run and per-phase cost rollups from the llm_calls view."""
        limit = max(1, min(limit, 200))
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id, call_count, total_cost_usd,
                           input_tokens, output_tokens,
                           cache_read_tokens, cache_creation_tokens
                    FROM v_cost_per_run
                    ORDER BY total_cost_usd DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                per_run = [dict(r) for r in cur.fetchall()]

                cur.execute(
                    """
                    SELECT phase, call_count, total_cost_usd,
                           input_tokens, output_tokens, avg_latency_seconds
                    FROM v_cost_per_phase
                    ORDER BY total_cost_usd DESC
                    """
                )
                per_phase = [dict(r) for r in cur.fetchall()]

                cur.execute(
                    """
                    SELECT model,
                           COUNT(*) AS calls,
                           COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens
                    FROM llm_calls
                    GROUP BY model
                    ORDER BY total_cost_usd DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                per_model = [dict(r) for r in cur.fetchall()]

        return {
            "per_run": per_run,
            "per_phase": per_phase,
            "per_model": per_model,
        }

    def query_throughput(self, *, days: int = 30) -> list[dict[str, Any]]:
        days = max(1, min(days, 365))
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT day, runs, success_runs, partial_runs, error_runs,
                           avg_pass_rate, avg_duration_seconds
                    FROM v_runs_per_day
                    LIMIT %s
                    """,
                    (days,),
                )
                rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            rr = dict(r)
            if rr.get("day") is not None:
                rr["day"] = str(rr["day"])
            out.append(rr)
        return out

    def query_quality(self) -> dict[str, Any]:
        """Derived quality metrics: first-attempt pass rate, mean attempts,
        refinement savings, pass-rate per metric."""
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT metric_name, total, passed, pass_rate,
                           avg_score, min_score, max_score
                    FROM v_pass_rate_per_metric
                    """
                )
                per_metric = [dict(r) for r in cur.fetchall()]

                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total_requirements,
                        COUNT(*) FILTER (WHERE ever_passed) AS passed_requirements,
                        COUNT(*) FILTER (WHERE ever_passed AND first_pass_attempt = 1)
                            AS first_attempt_passes,
                        AVG(final_attempt) FILTER (WHERE ever_passed)
                            AS mean_attempts_to_pass,
                        MAX(final_attempt) AS max_attempt
                    FROM v_attempts_per_requirement
                    """
                )
                attempts_row = cur.fetchone() or {}

        total_reqs = attempts_row.get("total_requirements") or 0
        passed_reqs = attempts_row.get("passed_requirements") or 0
        first_pass = attempts_row.get("first_attempt_passes") or 0
        return {
            "per_metric": per_metric,
            "total_requirements": total_reqs,
            "passed_requirements": passed_reqs,
            "first_attempt_pass_rate": (
                first_pass / total_reqs if total_reqs else 0.0
            ),
            "mean_attempts_to_pass": attempts_row.get("mean_attempts_to_pass"),
            "max_attempt": attempts_row.get("max_attempt"),
        }

    def query_incidents(
        self,
        *,
        limit: int = 50,
        category: str | None = None,
        unresolved_only: bool = False,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        where: list[str] = []
        params: list[Any] = []
        if category:
            where.append("category = %s")
            params.append(category)
        if unresolved_only:
            where.append("NOT resolved")
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        params.append(limit)
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, run_id, category, severity, message,
                           stack, phase, feature, resolved, resolved_at, timestamp
                    FROM incidents
                    {where_sql}
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        return [_stringify_timestamps(r) for r in rows]

    def query_agent_stats(
        self, *, run_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                if run_id:
                    cur.execute(
                        """
                        SELECT run_id, feature, agent,
                               SUM(activations) AS activations,
                               SUM(tool_calls) AS tool_calls,
                               SUM(decisions) AS decisions,
                               SUM(handoffs_in) AS handoffs_in,
                               SUM(handoffs_out) AS handoffs_out,
                               SUM(total_time_seconds) AS total_time_seconds
                        FROM agent_stats
                        WHERE run_id = %s
                        GROUP BY run_id, feature, agent
                        ORDER BY activations DESC
                        LIMIT %s
                        """,
                        (run_id, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT agent,
                               SUM(activations) AS activations,
                               SUM(tool_calls) AS tool_calls,
                               SUM(decisions) AS decisions,
                               SUM(handoffs_in) AS handoffs_in,
                               SUM(handoffs_out) AS handoffs_out,
                               SUM(total_time_seconds) AS total_time_seconds
                        FROM agent_stats
                        GROUP BY agent
                        ORDER BY activations DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def query_tool_calls(
        self, *, group_by: str = "tool", limit: int = 50
    ) -> list[dict[str, Any]]:
        if group_by not in {"tool", "agent", "feature"}:
            group_by = "tool"
        limit = max(1, min(limit, 500))
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {group_by} AS bucket,
                           COUNT(*) AS calls,
                           COUNT(*) FILTER (WHERE success) AS successes,
                           COUNT(*) FILTER (WHERE success IS FALSE) AS failures,
                           AVG(latency_seconds) FILTER (WHERE latency_seconds IS NOT NULL)
                               AS avg_latency_seconds
                    FROM tool_calls
                    GROUP BY {group_by}
                    ORDER BY calls DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def query_memory_activity(self) -> dict[str, Any]:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT operation,
                           COUNT(*) AS count,
                           AVG(latency_seconds) AS avg_latency_seconds
                    FROM memory_operations
                    GROUP BY operation
                    ORDER BY count DESC
                    """
                )
                per_operation = [dict(r) for r in cur.fetchall()]

                cur.execute(
                    """
                    SELECT memory_type, operation, COUNT(*) AS count
                    FROM memory_operations
                    WHERE memory_type IS NOT NULL
                    GROUP BY memory_type, operation
                    ORDER BY count DESC
                    """
                )
                per_type = [dict(r) for r in cur.fetchall()]

                cur.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE operation = 'recall' AND COALESCE(count, 0) > 0)
                            AS recall_hits,
                        COUNT(*) FILTER (WHERE operation = 'recall' AND COALESCE(count, 0) = 0)
                            AS recall_misses,
                        COUNT(*) FILTER (WHERE operation = 'create') AS created,
                        COUNT(*) FILTER (WHERE operation = 'boost') AS boosts,
                        COUNT(*) FILTER (WHERE operation = 'demote') AS demotes
                    FROM memory_operations
                    """
                )
                summary_row = cur.fetchone() or {}

        return {
            "per_operation": per_operation,
            "per_type": per_type,
            "summary": summary_row,
        }

    def query_decomposition(self, *, limit: int = 100) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS requirements_planned,
                        COALESCE(SUM(planned_sub_specs_count), 0) AS total_sub_specs,
                        AVG(planned_sub_specs_count) AS avg_sub_specs,
                        COUNT(*) FILTER (WHERE fallback) AS fallback_count,
                        COUNT(*) FILTER (WHERE empty_result) AS empty_result_count,
                        COUNT(*) FILTER (WHERE truncated) AS truncated_count,
                        COALESCE(SUM(depends_on_declared), 0) AS depends_on_declared,
                        COALESCE(SUM(depends_on_resolved), 0) AS depends_on_resolved,
                        COALESCE(SUM(depends_on_unresolved), 0) AS depends_on_unresolved
                    FROM decomposition_stats
                    """
                )
                summary_row = cur.fetchone() or {}

                cur.execute(
                    """
                    SELECT run_id, requirement_id, requirement_title,
                           planned_sub_specs_count, fallback, empty_result, truncated,
                           depends_on_declared, depends_on_resolved, depends_on_unresolved,
                           timestamp
                    FROM decomposition_stats
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = [_stringify_timestamps(r) for r in cur.fetchall()]

        return {"summary": summary_row, "rows": rows}

    def query_background_loop(self, *, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 2000))
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT active_task_count, pending_task_count,
                           completed_task_count, loop_restarts, timestamp
                    FROM background_loop_samples
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [_stringify_timestamps(r) for r in rows]

    def query_artifacts(self) -> dict[str, Any]:
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS files_written,
                        COALESCE(SUM(bytes_written), 0) AS total_bytes,
                        COUNT(*) FILTER (WHERE is_test) AS test_files,
                        COUNT(*) FILTER (WHERE NOT is_test) AS code_files
                    FROM artifact_writes
                    """
                )
                summary_row = cur.fetchone() or {}

                cur.execute(
                    """
                    SELECT language, COUNT(*) AS files,
                           COALESCE(SUM(bytes_written), 0) AS total_bytes
                    FROM artifact_writes
                    WHERE language IS NOT NULL
                    GROUP BY language
                    ORDER BY files DESC
                    """
                )
                per_language = [dict(r) for r in cur.fetchall()]

                cur.execute(
                    """
                    SELECT feature, COUNT(*) AS files,
                           COUNT(*) FILTER (WHERE is_test) AS test_files,
                           COALESCE(SUM(bytes_written), 0) AS total_bytes
                    FROM artifact_writes
                    WHERE feature IS NOT NULL
                    GROUP BY feature
                    ORDER BY files DESC
                    LIMIT 50
                    """
                )
                per_feature = [dict(r) for r in cur.fetchall()]

        return {
            "summary": summary_row,
            "per_language": per_language,
            "per_feature": per_feature,
        }

    def query_run_detail(self, run_id: str) -> dict[str, Any] | None:
        """Aggregate per-run detail for the run detail popup.

        Returns ``None`` if the run doesn't exist in ``pipeline_runs``.
        All sub-queries are scoped to a single ``run_id`` so the payload
        is safe to return in full — no global aggregates leak in.
        """
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                # 1) Core run row (must exist)
                cur.execute(
                    """
                    SELECT run_id, started_at, ended_at, status,
                           spec_count, feature_count, pass_rate,
                           duration_seconds, error, metadata
                    FROM pipeline_runs
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
                run_row = cur.fetchone()
                if run_row is None:
                    return None
                run_detail = _stringify_timestamps(run_row)

                # 2) LLM usage totals for this run
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total_calls,
                        COALESCE(SUM(input_tokens), 0) AS input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS output_tokens,
                        COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_tokens,
                        COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
                        AVG(latency_seconds) FILTER (WHERE latency_seconds IS NOT NULL) AS avg_latency_seconds,
                        COUNT(*) FILTER (WHERE rate_limited) AS rate_limited_count,
                        COUNT(*) FILTER (WHERE error IS NOT NULL) AS error_count
                    FROM llm_calls
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
                llm_totals = cur.fetchone() or {}

                # 3) LLM per-phase breakdown
                cur.execute(
                    """
                    SELECT phase, COUNT(*) AS calls,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens,
                           COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
                           AVG(latency_seconds) FILTER (WHERE latency_seconds IS NOT NULL) AS avg_latency_seconds
                    FROM llm_calls
                    WHERE run_id = %s
                    GROUP BY phase
                    ORDER BY calls DESC
                    """,
                    (run_id,),
                )
                llm_per_phase = [dict(r) for r in cur.fetchall()]

                # 4) Swarm feature events for this run
                cur.execute(
                    """
                    SELECT run_id, feature, event, status,
                           artifact_count, test_count, handoff_count,
                           layer, error, duration_seconds,
                           agent_transitions, unique_agents_visited,
                           planner_calls, coder_calls, reviewer_calls, tester_calls,
                           tool_call_count, tool_failure_count,
                           deep_agent_invocations, deep_agent_timeout_count,
                           worker_crash_count, timestamp
                    FROM swarm_feature_events
                    WHERE run_id = %s
                    ORDER BY timestamp ASC
                    """,
                    (run_id,),
                )
                swarm_events = [_stringify_timestamps(r) for r in cur.fetchall()]

                # 5) Agent stats for this run
                cur.execute(
                    """
                    SELECT agent,
                           SUM(activations) AS activations,
                           SUM(tool_calls) AS tool_calls,
                           SUM(decisions) AS decisions,
                           SUM(handoffs_in) AS handoffs_in,
                           SUM(handoffs_out) AS handoffs_out,
                           SUM(total_time_seconds) AS total_time_seconds
                    FROM agent_stats
                    WHERE run_id = %s
                    GROUP BY agent
                    ORDER BY activations DESC
                    """,
                    (run_id,),
                )
                agent_stats = [dict(r) for r in cur.fetchall()]

                # 6) Tool call breakdown grouped by tool name
                cur.execute(
                    """
                    SELECT tool AS bucket,
                           COUNT(*) AS calls,
                           COUNT(*) FILTER (WHERE success) AS successes,
                           COUNT(*) FILTER (WHERE success IS FALSE) AS failures,
                           AVG(latency_seconds) FILTER (WHERE latency_seconds IS NOT NULL) AS avg_latency_seconds
                    FROM tool_calls
                    WHERE run_id = %s
                    GROUP BY tool
                    ORDER BY calls DESC
                    """,
                    (run_id,),
                )
                tool_calls = [dict(r) for r in cur.fetchall()]

                # 7) Incidents for this run
                cur.execute(
                    """
                    SELECT id, run_id, category, severity, message,
                           stack, phase, feature, resolved, resolved_at, timestamp
                    FROM incidents
                    WHERE run_id = %s
                    ORDER BY timestamp DESC
                    """,
                    (run_id,),
                )
                incidents = [_stringify_timestamps(r) for r in cur.fetchall()]

                # 8) Eval metrics for this run
                cur.execute(
                    """
                    SELECT timestamp, metric_name, score, passed, attempt,
                           run_id, spec_id, requirement_id, eval_type, reason
                    FROM eval_metrics
                    WHERE run_id = %s
                    ORDER BY timestamp ASC
                    """,
                    (run_id,),
                )
                eval_metrics = [_stringify_timestamps(r) for r in cur.fetchall()]

                # 9) Artifact writes for this run (aggregated)
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS files_written,
                        COALESCE(SUM(bytes_written), 0) AS total_bytes,
                        COUNT(*) FILTER (WHERE is_test) AS test_files,
                        COUNT(*) FILTER (WHERE NOT is_test) AS code_files
                    FROM artifact_writes
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
                artifacts_summary = cur.fetchone() or {}

                cur.execute(
                    """
                    SELECT language, COUNT(*) AS files,
                           COALESCE(SUM(bytes_written), 0) AS total_bytes
                    FROM artifact_writes
                    WHERE run_id = %s AND language IS NOT NULL
                    GROUP BY language
                    ORDER BY files DESC
                    """,
                    (run_id,),
                )
                artifacts_per_language = [dict(r) for r in cur.fetchall()]

                # 10) Decomposition planning rows for this run
                cur.execute(
                    """
                    SELECT run_id, requirement_id, requirement_title,
                           planned_sub_specs_count, fallback, empty_result,
                           truncated, depends_on_declared, depends_on_resolved,
                           depends_on_unresolved, timestamp
                    FROM decomposition_stats
                    WHERE run_id = %s
                    ORDER BY timestamp ASC
                    """,
                    (run_id,),
                )
                decomposition = [_stringify_timestamps(r) for r in cur.fetchall()]

                # 11) Full progress event log for this run — the raw
                # append-only audit trail of every broker event emitted
                # during the pipeline. Powers the "Agent Log" tab in
                # the Run Detail popup so operators can replay the
                # exact sequence of agent actions / tool calls / eval
                # rubrics / lifecycle transitions that occurred.
                cur.execute(
                    """
                    SELECT id, run_id, event, feature, agent,
                           timestamp, payload
                    FROM progress_events
                    WHERE run_id = %s
                    ORDER BY timestamp ASC, id ASC
                    """,
                    (run_id,),
                )
                progress_log = [_stringify_timestamps(r) for r in cur.fetchall()]

        return {
            "run": run_detail,
            "llm": {
                "totals": llm_totals,
                "per_phase": llm_per_phase,
            },
            "swarm_events": swarm_events,
            "agent_stats": agent_stats,
            "tool_calls": tool_calls,
            "incidents": incidents,
            "eval_metrics": eval_metrics,
            "artifacts": {
                "summary": artifacts_summary,
                "per_language": artifacts_per_language,
            },
            "decomposition": decomposition,
            "progress_log": progress_log,
        }


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _stringify_timestamps(row: dict[str, Any]) -> dict[str, Any]:
    """Convert datetime fields to ISO strings for JSON serialisation."""
    import datetime

    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, datetime.datetime):
            out[k] = v.isoformat()
        elif isinstance(v, datetime.date):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out
