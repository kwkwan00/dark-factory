"""Postgres writer for refinery v2 forensic tables.

Sibling to ``MetricsRepository``. The dual-write contract is load-bearing
for existing cost/usage dashboards: every refinery LLM call lands in
both ``llm_calls`` (existing table, with ``phase='refinery.{role}.{kind}'``)
and ``refinery_llm_calls`` (new table, with role/requirement/round/tool-
calls columns that ``llm_calls`` doesn't carry). Both INSERTs run inside
one Postgres transaction — either-side failure rolls the whole thing
back. A dual-write failure is recorded on
``dark_factory_refinery_dual_write_failures_total{table}`` but never
fails the debate itself; telemetry loss is always preferable to a
telemetry-driven outage.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import structlog
from psycopg.types.json import Json

from dark_factory.log import trace_methods
from dark_factory.metrics.prometheus import (
    observe_refinery_dual_write_failure,
    observe_refinery_postgres_write,
)

log = structlog.get_logger()


@trace_methods
class RefineryMetricsRepository:
    """Read/write API over the refinery_* forensic tables."""

    def __init__(self, client) -> None:
        self.client = client

    # ── Writes: run lifecycle ─────────────────────────────────────────────

    def record_run_start(
        self,
        *,
        refinery_run_id: str,
        source_mode: str,
        source_run_id: str | None = None,
        requirements_count: int = 0,
        settings_snapshot: dict[str, Any] | None = None,
    ) -> None:
        t0 = time.monotonic()
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO refinery_runs (
                        refinery_run_id, source_mode, source_run_id,
                        requirements_count, settings_snapshot
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (refinery_run_id) DO UPDATE SET
                        source_mode = EXCLUDED.source_mode,
                        source_run_id = EXCLUDED.source_run_id,
                        requirements_count = EXCLUDED.requirements_count,
                        settings_snapshot = EXCLUDED.settings_snapshot
                    """,
                    (
                        refinery_run_id,
                        source_mode,
                        source_run_id,
                        requirements_count,
                        Json(settings_snapshot or {}),
                    ),
                )
            conn.commit()
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)

    def record_run_end(
        self,
        *,
        refinery_run_id: str,
        converged_count: int,
        short_circuited_count: int,
        aborted_count: int,
        duration_seconds: float | None,
        total_cost_usd: float,
        total_tokens_in: int,
        total_tokens_out: int,
    ) -> None:
        t0 = time.monotonic()
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE refinery_runs
                       SET ended_at = NOW(),
                           converged_count = %s,
                           short_circuited_count = %s,
                           aborted_count = %s,
                           duration_seconds = %s,
                           total_cost_usd = %s,
                           total_tokens_in = %s,
                           total_tokens_out = %s
                     WHERE refinery_run_id = %s
                    """,
                    (
                        converged_count,
                        short_circuited_count,
                        aborted_count,
                        duration_seconds,
                        total_cost_usd,
                        total_tokens_in,
                        total_tokens_out,
                        refinery_run_id,
                    ),
                )
            conn.commit()
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)

    # ── Resume support (V1 — requirement-level) ──────────────────────────

    def record_run_status(
        self,
        *,
        refinery_run_id: str,
        status: str,
    ) -> None:
        """Update the run's lifecycle status.

        Status transitions:
        - ``in_progress`` → set at run start.
        - ``completed`` → set at the Phase 4 done-event.
        - ``cancelled`` → set when the SSE consumer disconnects mid-run.
        - ``failed`` → set on uncaught exception.
        """

        t0 = time.monotonic()
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE refinery_runs
                       SET status = %s
                     WHERE refinery_run_id = %s
                    """,
                    (status, refinery_run_id),
                )
            conn.commit()
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)

    # Window after which an ``in_progress`` row with no ``ended_at`` is
    # treated as orphaned (the worker died before the finally block
    # could flip status → failed). 15 minutes is comfortably longer
    # than typical refinery runs (2-5 min) so a healthy run can never
    # be mistaken for stale.
    _STALE_IN_PROGRESS_INTERVAL = "15 minutes"

    def try_claim_in_progress(self, refinery_run_id: str) -> bool:
        """Atomically transition a run to ``in_progress``, returning True
        iff this caller won the race. Succeeds either when the row is
        in a terminal state (``completed`` / ``failed`` / ``cancelled``)
        or when it's been stuck ``in_progress`` long enough that the
        original worker is presumed dead.

        Backstops two scenarios:

        1. **Double-click**: two concurrent POSTs to
           ``/refinery/resume/{id}`` both pass the load-status guard
           client-side but only one runs Phase 2. The row lock + the
           ``started_at = NOW()`` reset make the UPDATE single-winner
           — the second claimer's WHERE no longer matches the stale
           branch (started_at is fresh) nor the terminal branch (status
           is in_progress).
        2. **Hard-killed worker**: a container crash or `kill -9`
           leaves status at ``in_progress`` because the orchestrator's
           ``finally`` couldn't run. After
           ``_STALE_IN_PROGRESS_INTERVAL`` elapses, the row is
           reclaimable.
        """

        t0 = time.monotonic()
        try:
            with self.client.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        UPDATE refinery_runs
                           SET status = 'in_progress',
                               started_at = NOW(),
                               ended_at = NULL
                         WHERE refinery_run_id = %s
                           AND (
                             status != 'in_progress'
                             OR (status = 'in_progress'
                                 AND started_at < NOW()
                                                  - INTERVAL '{self._STALE_IN_PROGRESS_INTERVAL}')
                           )
                        RETURNING refinery_run_id
                        """,
                        (refinery_run_id,),
                    )
                    row = cur.fetchone()
                conn.commit()
            observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)
            return row is not None
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_try_claim_in_progress_failed", error=str(exc),
            )
            # Fall back to "claim succeeded" so a Postgres outage
            # doesn't block legitimate resumes; the worst case is the
            # rare double-click loses both signals.
            return True

    def record_input_snapshot(
        self,
        *,
        refinery_run_id: str,
        input_snapshot: dict[str, Any],
    ) -> None:
        """Persist the run's input payload so a resumed run can skip
        Phase 1 (gather). Snapshot includes the source mode, source
        run_id / input_path / direct payload, and the gathered
        requirements list."""

        t0 = time.monotonic()
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE refinery_runs
                       SET input_snapshot = %s
                     WHERE refinery_run_id = %s
                    """,
                    (Json(input_snapshot), refinery_run_id),
                )
            conn.commit()
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)

    def record_debate_completion(
        self,
        *,
        refinery_run_id: str,
        requirement_id: str,
        convergence_status: str,
        refined_payload: dict[str, Any],
        suggested_memories: list[dict[str, Any]] | None = None,
    ) -> None:
        """Cache a completed debate's output so resume can replay it."""

        t0 = time.monotonic()
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO refinery_debate_completions (
                        refinery_run_id, requirement_id, convergence_status,
                        refined_payload, suggested_memories
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (refinery_run_id, requirement_id) DO UPDATE SET
                        convergence_status = EXCLUDED.convergence_status,
                        refined_payload = EXCLUDED.refined_payload,
                        suggested_memories = EXCLUDED.suggested_memories,
                        completed_at = NOW()
                    """,
                    (
                        refinery_run_id,
                        requirement_id,
                        convergence_status,
                        Json(refined_payload),
                        Json(suggested_memories or []),
                    ),
                )
            conn.commit()
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)

    def load_run_for_resume(
        self,
        refinery_run_id: str,
    ) -> dict[str, Any] | None:
        """Load the resume-relevant state for one run.

        Returns ``None`` when the run is unknown to Postgres. The
        returned dict carries: ``status``, ``source_mode``, ``source_run_id``,
        ``input_snapshot``, ``requirements_count``, ``started_at``.
        """

        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT status, source_mode, source_run_id,
                           input_snapshot, requirements_count, started_at
                      FROM refinery_runs
                     WHERE refinery_run_id = %s
                    """,
                    (refinery_run_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return {
            "status": row["status"],
            "source_mode": row["source_mode"],
            "source_run_id": row["source_run_id"],
            "input_snapshot": row["input_snapshot"] or {},
            "requirements_count": row["requirements_count"],
            "started_at": row["started_at"],
        }

    def load_completed_debates(
        self,
        refinery_run_id: str,
    ) -> dict[str, dict[str, Any]]:
        """Return ``{requirement_id: {convergence_status, refined,
        suggested_memories, completed_at}}`` for every cached
        completion of ``refinery_run_id``."""

        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT requirement_id, convergence_status,
                           refined_payload, suggested_memories,
                           completed_at
                      FROM refinery_debate_completions
                     WHERE refinery_run_id = %s
                    """,
                    (refinery_run_id,),
                )
                rows = cur.fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            out[r["requirement_id"]] = {
                "convergence_status": r["convergence_status"],
                "refined": r["refined_payload"],
                "suggested_memories": r["suggested_memories"] or [],
                "completed_at": r["completed_at"],
            }
        return out

    # ── Lifecycle helpers ────────────────────────────────────────────────

    def record_applied_to_graph(
        self,
        *,
        refinery_run_id: str,
    ) -> None:
        t0 = time.monotonic()
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE refinery_runs
                       SET applied_to_graph = TRUE,
                           applied_at = NOW()
                     WHERE refinery_run_id = %s
                    """,
                    (refinery_run_id,),
                )
            conn.commit()
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)

    # ── Writes: debate lifecycle ──────────────────────────────────────────

    def record_debate(
        self,
        *,
        refinery_run_id: str,
        requirement_id: str,
        convergence_status: str,
        rounds_executed: int,
        escalation_level: int = 0,
        research_calls_used: int = 0,
        final_overall_score: float | None = None,
        final_dimension_scores: dict[str, float] | None = None,
        disagreement_score_max: float | None = None,
        duration_seconds: float | None = None,
        cost_usd: float = 0.0,
        termination_reason: str | None = None,
        reconcile_invoked: bool = False,
        priority: str | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """Insert one debate row and return its generated id."""

        t0 = time.monotonic()
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO refinery_debates (
                        refinery_run_id, requirement_id, convergence_status,
                        final_overall_score, final_dimension_scores,
                        rounds_executed, escalation_level, research_calls_used,
                        disagreement_score_max, duration_seconds, cost_usd,
                        termination_reason, reconcile_invoked,
                        priority, tags
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        refinery_run_id,
                        requirement_id,
                        convergence_status,
                        final_overall_score,
                        Json(final_dimension_scores or {}),
                        rounds_executed,
                        escalation_level,
                        research_calls_used,
                        disagreement_score_max,
                        duration_seconds,
                        cost_usd,
                        termination_reason,
                        reconcile_invoked,
                        priority,
                        Json(list(tags or [])),
                    ),
                )
                debate_id = cur.fetchone()["id"]
            conn.commit()
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)
        return int(debate_id)

    def record_round(
        self,
        *,
        debate_id: int,
        round_number: int,
        critic_count: int,
        critic_blockers_count: int,
        critic_warnings_count: int,
        rule_violations_count: int,
        rule_warnings_count: int,
        judge_overall: float | None,
        judge_dimensions: dict[str, float] | None,
        disagreement_score: float | None,
        router_decision: str,
        duration_seconds: float | None = None,
    ) -> None:
        t0 = time.monotonic()
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO refinery_debate_rounds (
                        debate_id, round_number, critic_count,
                        critic_blockers_count, critic_warnings_count,
                        rule_violations_count, rule_warnings_count,
                        judge_overall, judge_dimensions,
                        disagreement_score, router_decision, duration_seconds
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (debate_id, round_number) DO UPDATE SET
                        critic_count = EXCLUDED.critic_count,
                        critic_blockers_count = EXCLUDED.critic_blockers_count,
                        critic_warnings_count = EXCLUDED.critic_warnings_count,
                        rule_violations_count = EXCLUDED.rule_violations_count,
                        rule_warnings_count = EXCLUDED.rule_warnings_count,
                        judge_overall = EXCLUDED.judge_overall,
                        judge_dimensions = EXCLUDED.judge_dimensions,
                        disagreement_score = EXCLUDED.disagreement_score,
                        router_decision = EXCLUDED.router_decision,
                        duration_seconds = EXCLUDED.duration_seconds
                    """,
                    (
                        debate_id,
                        round_number,
                        critic_count,
                        critic_blockers_count,
                        critic_warnings_count,
                        rule_violations_count,
                        rule_warnings_count,
                        judge_overall,
                        Json(judge_dimensions or {}),
                        disagreement_score,
                        router_decision,
                        duration_seconds,
                    ),
                )
            conn.commit()
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)

    # ── Writes: LLM call (DUAL-WRITE to llm_calls + refinery_llm_calls) ──

    def record_llm_call(
        self,
        *,
        refinery_run_id: str,
        role: str,
        kind: str,
        model: str,
        requirement_id: str | None = None,
        round_number: int | None = None,
        reasoning_effort: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cache_read_tokens: int | None = None,
        latency_ms: int | None = None,
        cost_usd: float | None = None,
        error: str | None = None,
        tool_calls: list[dict] | None = None,
        started_at: datetime | None = None,
    ) -> None:
        """DUAL-WRITE: one INSERT into ``llm_calls`` (so existing cost
        dashboards see refinery) + one INSERT into ``refinery_llm_calls``
        (with role + requirement + round + tool_calls), inside a single
        transaction. Either-side failure rolls the whole thing back and
        increments the ``refinery_dual_write_failures_total{table}``
        counter — the debate continues.
        """

        phase = f"refinery.{role}.{kind}"
        latency_seconds = (latency_ms or 0) / 1000.0 if latency_ms else None
        t0 = time.monotonic()
        failing_table = ""
        try:
            with self.client.connection() as conn:
                with conn.cursor() as cur:
                    # Dual-write #1: existing llm_calls (canonical system-wide
                    # record used by all existing cost / usage dashboards).
                    failing_table = "llm_calls"
                    cur.execute(
                        """
                        INSERT INTO llm_calls (
                            run_id, client, model, phase,
                            input_tokens, output_tokens,
                            cache_read_input_tokens,
                            latency_seconds, cost_usd, error
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            refinery_run_id,
                            "openai" if model.startswith(("gpt-", "o-")) else "anthropic",
                            model,
                            phase,
                            tokens_in,
                            tokens_out,
                            cache_read_tokens,
                            latency_seconds,
                            cost_usd,
                            error,
                        ),
                    )
                    # Dual-write #2: refinery-specific table with the full
                    # structured context the legacy table can't carry.
                    failing_table = "refinery_llm_calls"
                    cur.execute(
                        """
                        INSERT INTO refinery_llm_calls (
                            refinery_run_id, requirement_id, round_number,
                            role, kind, model, reasoning_effort,
                            tokens_in, tokens_out, cache_read_tokens,
                            latency_ms, cost_usd, started_at, error, tool_calls
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, COALESCE(%s, NOW()), %s, %s)
                        """,
                        (
                            refinery_run_id,
                            requirement_id,
                            round_number,
                            role,
                            kind,
                            model,
                            reasoning_effort,
                            tokens_in,
                            tokens_out,
                            cache_read_tokens,
                            latency_ms,
                            cost_usd,
                            started_at,
                            error,
                            Json(tool_calls or []),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            # Transaction already rolled back by psycopg when commit()
            # didn't land — either INSERT rolled both back. Record which
            # side failed first and move on; never raise.
            observe_refinery_dual_write_failure(table=failing_table or "unknown")
            log.warning(
                "refinery_llm_call_dual_write_failed",
                failing_table=failing_table,
                error=str(exc),
            )
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)

    # ── Writes: rule violations, research sources, memory audits ─────────

    def record_rule_violation(
        self,
        *,
        debate_id: int,
        round_number: int,
        rule_id: str,
        severity: str,
        dimension: str,
        finding: str,
        suggested_fix: str | None = None,
        injected_as_critique: bool = False,
    ) -> None:
        t0 = time.monotonic()
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO refinery_rule_violations (
                        debate_id, round_number, rule_id, severity, dimension,
                        finding, suggested_fix, injected_as_critique
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        debate_id,
                        round_number,
                        rule_id,
                        severity,
                        dimension,
                        finding,
                        suggested_fix,
                        injected_as_critique,
                    ),
                )
            conn.commit()
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)

    def record_research_source(
        self,
        *,
        debate_id: int,
        round_number: int,
        tier: int,
        provider: str,
        url: str | None,
        title: str | None,
        propagated: bool,
        insight_confidence: float | None = None,
    ) -> None:
        t0 = time.monotonic()
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO refinery_research_sources (
                        debate_id, round_number, tier, provider, url, title,
                        propagated, insight_confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        debate_id,
                        round_number,
                        tier,
                        provider,
                        url,
                        title,
                        propagated,
                        insight_confidence,
                    ),
                )
            conn.commit()
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)

    def compute_provider_stats(
        self,
        *,
        window_days: int = 30,
    ) -> list[dict[str, Any]]:
        """Return aggregated contribution stats per ``(tier, provider)``
        over the last ``window_days``.

        Each row carries:

        - ``tier`` (int 0–5)
        - ``provider`` (provider class id, e.g. ``official_docs_t3``)
        - ``calls`` (rows in the window)
        - ``propagated`` (rows with ``propagated=true``)
        - ``rate`` (propagated / calls; 0 when calls == 0)
        - ``mean_confidence`` (avg ``insight_confidence`` over
          propagated rows; ``None`` when no propagations)

        The T3/T5 provider learning loop reads this to decide whether
        to soft-demote or boost a provider's trust weight on the next
        run.
        """

        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        tier,
                        provider,
                        COUNT(*) AS calls,
                        SUM(CASE WHEN propagated THEN 1 ELSE 0 END) AS propagated,
                        AVG(CASE WHEN propagated THEN insight_confidence ELSE NULL END)
                            AS mean_confidence
                      FROM refinery_research_sources
                     WHERE fetched_at >= NOW() - (%s || ' days')::interval
                     GROUP BY tier, provider
                     ORDER BY tier, provider
                    """,
                    (window_days,),
                )
                rows = cur.fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            tier = row["tier"]
            provider = row["provider"]
            calls = int(row["calls"] or 0)
            propagated = int(row["propagated"] or 0)
            mean_conf = row["mean_confidence"]
            rate = (propagated / calls) if calls else 0.0
            out.append({
                "tier": int(tier),
                "provider": provider,
                "calls": calls,
                "propagated": propagated,
                "rate": rate,
                "mean_confidence": (
                    float(mean_conf) if mean_conf is not None else None
                ),
            })
        return out

    def record_memory_audit(
        self,
        *,
        refinery_run_id: str,
        suggested_memory_id: str,
        kind: str,
        source_role: str,
        validation_status: str,
        outcome: str,
        debate_id: int | None = None,
        existing_memory_id: str | None = None,
        similarity: float | None = None,
        provenance_source_tier_mix: list[int] | None = None,
        provenance_confidence: float | None = None,
    ) -> None:
        t0 = time.monotonic()
        with self.client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO refinery_memory_audits (
                        debate_id, refinery_run_id, suggested_memory_id,
                        kind, source_role, validation_status, outcome,
                        existing_memory_id, similarity,
                        provenance_source_tier_mix, provenance_confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        debate_id,
                        refinery_run_id,
                        suggested_memory_id,
                        kind,
                        source_role,
                        validation_status,
                        outcome,
                        existing_memory_id,
                        similarity,
                        provenance_source_tier_mix or [],
                        provenance_confidence,
                    ),
                )
            conn.commit()
        observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)

    # ── Active-learning: memory feedback + role weighting ────────────────

    def record_memory_feedback(
        self,
        *,
        memory_id: str,
        memory_kind: str,
        decision: str,
        refinery_run_id: str | None = None,
        source_role: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Record one operator decision (save/dismiss/edit) on a memory.

        Best-effort: a Postgres outage logs and continues so the UI
        action never fails because telemetry is unhappy.
        """

        t0 = time.monotonic()
        try:
            with self.client.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO refinery_memory_feedback (
                            refinery_run_id, memory_id, memory_kind,
                            source_role, decision, reason
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            refinery_run_id,
                            memory_id,
                            memory_kind,
                            source_role,
                            decision,
                            reason,
                        ),
                    )
                conn.commit()
            observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("refinery_memory_feedback_write_failed", error=str(exc))

    def record_critique_dispositions(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        """Batch-insert disposition rows in one round-trip.

        Each row dict carries: refinery_run_id, requirement_id,
        round_number, role, severity, action, optional dimension.
        Best-effort — Postgres outage logs and continues.
        """

        if not rows:
            return
        t0 = time.monotonic()
        try:
            with self.client.connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO refinery_critique_dispositions (
                            refinery_run_id, requirement_id, round_number,
                            role, severity, dimension, action
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                r["refinery_run_id"],
                                r["requirement_id"],
                                r["round_number"],
                                r["role"],
                                r["severity"],
                                r.get("dimension"),
                                r["action"],
                            )
                            for r in rows
                        ],
                    )
                conn.commit()
            observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_critique_disposition_batch_write_failed",
                error=str(exc), count=len(rows),
            )

    def record_critique_disposition(
        self,
        *,
        refinery_run_id: str,
        requirement_id: str,
        round_number: int,
        role: str,
        severity: str,
        action: str,
        dimension: str | None = None,
    ) -> None:
        """Record one (round, critique) disposition for role-weighting.

        Single-row variant kept for tests and ad-hoc callers; production
        path uses ``record_critique_dispositions`` for batched insert.
        """

        t0 = time.monotonic()
        try:
            with self.client.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO refinery_critique_dispositions (
                            refinery_run_id, requirement_id, round_number,
                            role, severity, dimension, action
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            refinery_run_id,
                            requirement_id,
                            round_number,
                            role,
                            severity,
                            dimension,
                            action,
                        ),
                    )
                conn.commit()
            observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_critique_disposition_write_failed", error=str(exc)
            )

    def record_requirement_decision(
        self,
        *,
        refinery_run_id: str,
        requirement_id: str,
        decision: str,
        judge_overall_score: float | None = None,
        convergence_status: str | None = None,
    ) -> None:
        """Record one operator decision on a refined requirement.

        Best-effort: Postgres outage logs and continues so the UI
        action never fails because telemetry is unhappy.
        """

        t0 = time.monotonic()
        try:
            with self.client.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO refinery_requirement_decisions (
                            refinery_run_id, requirement_id, decision,
                            judge_overall_score, convergence_status
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            refinery_run_id,
                            requirement_id,
                            decision,
                            judge_overall_score,
                            convergence_status,
                        ),
                    )
                conn.commit()
            observe_refinery_postgres_write(duration_seconds=time.monotonic() - t0)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_requirement_decision_write_failed", error=str(exc),
            )

    def compute_judge_calibration_stats(
        self, *, window_days: int = 30, score_split: float = 0.8,
    ) -> dict[str, Any]:
        """Aggregate operator decisions vs. Judge overall scores over
        the trailing window.

        Returns four counts that drive the calibration multiplier:

        - ``high_applied``     — Judge said "pass" + operator applied  (calibrated)
        - ``high_dismissed``   — Judge said "pass" + operator dismissed (overconfident)
        - ``low_applied``      — Judge said "fail" + operator applied   (underconfident)
        - ``low_dismissed``    — Judge said "fail" + operator dismissed (calibrated)

        The split point defaults to the configured pass threshold
        (0.8). Edited decisions count as half-applied / half-dismissed
        — a softer signal than a clean dismissal.

        Empty stats when Postgres is unavailable or no rows exist —
        callers fall through to identity multiplier.
        """

        try:
            with self.client.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT decision, judge_overall_score, convergence_status
                        FROM refinery_requirement_decisions
                        WHERE decided_at > NOW() - make_interval(days => %s)
                          AND (judge_overall_score IS NOT NULL
                               OR convergence_status IS NOT NULL)
                        """,
                        (int(window_days),),
                    )
                    rows = cur.fetchall()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_calibration_query_failed", error=str(exc),
            )
            return {}

        high_applied = high_dismissed = 0.0
        low_applied = low_dismissed = 0.0
        for row in rows:
            decision = row["decision"]
            score = row["judge_overall_score"]
            convergence = row["convergence_status"]
            # Prefer the explicit score. When absent (frontend hasn't
            # plumbed it through yet), fall back to convergence_status
            # as a coarse high/low proxy: converged → high; short-
            # circuited / aborted → low.
            if score is not None:
                high = float(score) >= score_split
            elif convergence == "converged":
                high = True
            else:
                high = False
            if decision == "accepted":
                if high:
                    high_applied += 1
                else:
                    low_applied += 1
            elif decision == "dismissed":
                if high:
                    high_dismissed += 1
                else:
                    low_dismissed += 1
            elif decision == "edited":
                # Half-credit on each side of the binary.
                if high:
                    high_applied += 0.5
                    high_dismissed += 0.5
                else:
                    low_applied += 0.5
                    low_dismissed += 0.5

        return {
            "high_applied": high_applied,
            "high_dismissed": high_dismissed,
            "low_applied": low_applied,
            "low_dismissed": low_dismissed,
            "total": float(len(rows)),
            "score_split": score_split,
            "window_days": window_days,
        }

    def compute_role_acceptance_stats(
        self, *, window_days: int = 30,
    ) -> list[dict[str, Any]]:
        """Aggregate per-role critique acceptance over the trailing
        window. Returns one row per (role, severity) with counts of
        accepted / rejected / deferred and the implied acceptance rate.
        """

        try:
            with self.client.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT role, severity,
                               COUNT(*) FILTER (WHERE action='accepted') AS accepted,
                               COUNT(*) FILTER (WHERE action='rejected') AS rejected,
                               COUNT(*) FILTER (WHERE action='deferred') AS deferred,
                               COUNT(*)                                   AS total
                        FROM refinery_critique_dispositions
                        WHERE recorded_at > NOW() - make_interval(days => %s)
                        GROUP BY role, severity
                        """,
                        (int(window_days),),
                    )
                    rows = cur.fetchall()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("refinery_role_stats_query_failed", error=str(exc))
            return []

        out: list[dict[str, Any]] = []
        for row in rows:
            t = int(row["total"] or 0)
            a = int(row["accepted"] or 0)
            out.append({
                "role": row["role"],
                "severity": row["severity"],
                "accepted": a,
                "rejected": int(row["rejected"] or 0),
                "deferred": int(row["deferred"] or 0),
                "total": t,
                "acceptance_rate": (a / t) if t else 0.0,
            })
        return out

    def compute_role_dimension_acceptance_stats(
        self, *, window_days: int = 30, severity: str = "blocker",
    ) -> list[dict[str, Any]]:
        """Aggregate per-(role, dimension) critique acceptance over the
        trailing window. Sibling to ``compute_role_acceptance_stats``,
        but split by ``dimension`` so a role's credibility on
        ``risk_coverage`` doesn't average together with its credibility
        on ``feasibility``.

        Rows where ``dimension`` is NULL are dropped (not all critiques
        carry a dimension; flat role-weighting handles those). Filtered
        by ``severity`` — accepting a WARNING is much cheaper than
        accepting a BLOCKER, so the production loop only learns from
        blockers (matches the flat role-weighting scope).
        """

        try:
            with self.client.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT role, dimension,
                               COUNT(*) FILTER (WHERE action='accepted') AS accepted,
                               COUNT(*) FILTER (WHERE action='rejected') AS rejected,
                               COUNT(*) FILTER (WHERE action='deferred') AS deferred,
                               COUNT(*)                                   AS total
                        FROM refinery_critique_dispositions
                        WHERE recorded_at > NOW() - make_interval(days => %s)
                          AND severity = %s
                          AND dimension IS NOT NULL
                        GROUP BY role, dimension
                        """,
                        (int(window_days), severity),
                    )
                    rows = cur.fetchall()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("refinery_role_dim_stats_query_failed", error=str(exc))
            return []

        out: list[dict[str, Any]] = []
        for row in rows:
            t = int(row["total"] or 0)
            a = int(row["accepted"] or 0)
            out.append({
                "role": row["role"],
                "dimension": row["dimension"],
                "severity": severity,
                "accepted": a,
                "rejected": int(row["rejected"] or 0),
                "deferred": int(row["deferred"] or 0),
                "total": t,
                "acceptance_rate": (a / t) if t else 0.0,
            })
        return out

    def compute_archetype_round_stats(
        self, *, window_days: int = 30,
    ) -> list[dict[str, Any]]:
        """Aggregate convergence behaviour per requirement archetype.

        Buckets are ``(priority, source_mode, primary_tag)`` where
        ``primary_tag`` is the first tag in the requirement's tag list
        ('untagged' when empty). For each bucket returns:

        - ``total`` debates in the window
        - ``converged`` / ``short_circuited`` / ``aborted`` counts
        - ``avg_rounds_converged`` (mean ``rounds_executed`` over
          converged debates only — short-circuits would skew the mean
          toward the cap)

        Identity defaults are applied upstream when ``total`` is below
        the learning threshold.
        """

        try:
            with self.client.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            COALESCE(d.priority, 'unknown')                 AS priority,
                            COALESCE(r.source_mode, 'unknown')              AS source_mode,
                            COALESCE(d.tags->>0, 'untagged')                AS primary_tag,
                            COUNT(*)                                        AS total,
                            COUNT(*) FILTER (WHERE d.convergence_status='converged')       AS converged,
                            COUNT(*) FILTER (WHERE d.convergence_status='short_circuited') AS short_circuited,
                            COUNT(*) FILTER (WHERE d.convergence_status='aborted')         AS aborted,
                            AVG(d.rounds_executed) FILTER (WHERE d.convergence_status='converged')
                                                                            AS avg_rounds_converged
                        FROM refinery_debates d
                        JOIN refinery_runs r USING (refinery_run_id)
                        WHERE r.started_at > NOW() - make_interval(days => %s)
                        GROUP BY priority, source_mode, primary_tag
                        """,
                        (int(window_days),),
                    )
                    rows = cur.fetchall()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("refinery_archetype_stats_query_failed", error=str(exc))
            return []

        out: list[dict[str, Any]] = []
        for row in rows:
            t = int(row["total"] or 0)
            c = int(row["converged"] or 0)
            sc = int(row["short_circuited"] or 0)
            avg_r = row["avg_rounds_converged"]
            out.append({
                "priority": row["priority"],
                "source_mode": row["source_mode"],
                "primary_tag": row["primary_tag"],
                "total": t,
                "converged": c,
                "short_circuited": sc,
                "aborted": int(row["aborted"] or 0),
                "short_circuit_rate": (sc / t) if t else 0.0,
                "avg_rounds_converged": float(avg_r) if avg_r is not None else None,
            })
        return out

    def compute_rule_override_stats(
        self, *, window_days: int = 30,
    ) -> list[dict[str, Any]]:
        """Aggregate per-rule operator-override behaviour for blockers.

        Joins ``refinery_rule_violations`` (severity=blocker) to the
        operator decision on the parent debate's requirement. A blocker
        that fires followed by an operator ``accepted`` decision is an
        **override** — the operator applied the requirement despite the
        rule blocking convergence. High override rates signal a rule
        that's too strict for the deployment and should be auto-demoted
        from blocker to warning.

        One row per rule_id with at least one blocker + decision in the
        window. Returns:

        - ``rule_id``
        - ``blockers`` (count of blocker firings)
        - ``decided`` (count of those that have an operator decision)
        - ``overrides`` (operator applied despite blocker)
        - ``dismissals`` (operator dismissed — rule was right)
        - ``override_rate`` (overrides / decided; 0 when decided == 0)

        Warnings are intentionally excluded — only blockers gate
        convergence, so only blockers are candidates for severity demotion.
        """

        try:
            with self.client.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        WITH blockers AS (
                            SELECT
                                v.rule_id,
                                d.refinery_run_id,
                                d.requirement_id
                            FROM refinery_rule_violations v
                            JOIN refinery_debates d ON v.debate_id = d.id
                            JOIN refinery_runs    r ON d.refinery_run_id = r.refinery_run_id
                            WHERE v.severity = 'blocker'
                              AND r.started_at > NOW() - make_interval(days => %s)
                        )
                        SELECT
                            b.rule_id,
                            COUNT(*)                                       AS blockers,
                            COUNT(rd.decision)                             AS decided,
                            COUNT(*) FILTER (WHERE rd.decision='accepted') AS overrides,
                            COUNT(*) FILTER (WHERE rd.decision='dismissed') AS dismissals
                        FROM blockers b
                        LEFT JOIN refinery_requirement_decisions rd
                               ON rd.refinery_run_id = b.refinery_run_id
                              AND rd.requirement_id  = b.requirement_id
                        GROUP BY b.rule_id
                        """,
                        (int(window_days),),
                    )
                    rows = cur.fetchall()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("refinery_rule_override_stats_query_failed", error=str(exc))
            return []

        out: list[dict[str, Any]] = []
        for row in rows:
            d = int(row["decided"] or 0)
            o = int(row["overrides"] or 0)
            out.append({
                "rule_id": row["rule_id"],
                "blockers": int(row["blockers"] or 0),
                "decided": d,
                "overrides": o,
                "dismissals": int(row["dismissals"] or 0),
                "override_rate": (o / d) if d else 0.0,
            })
        return out
