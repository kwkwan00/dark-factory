"""ObservabilityHub — single sink for refinery telemetry.

Fans refinery events out to the three stores defined in the plan's
"Data-store layering" section:

- **Prometheus** (always on) — via ``metrics/prometheus.py::observe_refinery_*``
- **Postgres** (optional, FastAPI lifespan-owned) — via
  ``RefineryMetricsRepository`` when ``settings.postgres.enabled`` and
  ``settings.pipeline.refinery_postgres_forensics_enabled``
- **Trace JSON** (always on) — via the active ``TraceContext``

Construction is cheap (no LLM calls, no DB probes) so FastAPI's lifespan
can build one per process unconditionally. A missing Postgres client
reduces the hub to Prometheus + trace only, with a single structured
log warning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from dark_factory.api.refinery.contracts import (
    MemoryKind,
    SourceTier,
    ValidationStatus,
)
from dark_factory.log import trace_methods
from dark_factory.metrics import prometheus as prom

if TYPE_CHECKING:
    from dark_factory.metrics.refinery_repository import RefineryMetricsRepository

log = structlog.get_logger()


@trace_methods
class ObservabilityHub:
    """Facade over Prometheus + Postgres + TraceContext."""

    def __init__(
        self,
        *,
        refinery_repo: "RefineryMetricsRepository | None" = None,
        postgres_enabled: bool = False,
        prometheus_enabled: bool = True,
    ) -> None:
        self._repo = refinery_repo
        self._postgres_enabled = postgres_enabled and refinery_repo is not None
        self._prometheus_enabled = prometheus_enabled
        if postgres_enabled and refinery_repo is None:
            log.warning(
                "refinery_observability_hub_missing_postgres",
                reason="repo not supplied; Postgres forensic writes disabled",
            )

    # ── Run lifecycle ─────────────────────────────────────────────────────

    def on_run_start(
        self,
        *,
        refinery_run_id: str,
        source_mode: str,
        source_run_id: str | None,
        requirements_count: int,
        settings_snapshot: dict[str, Any],
    ) -> None:
        log.info(
            "refinery_run_started",
            refinery_run_id=refinery_run_id,
            source_mode=source_mode,
            requirements_count=requirements_count,
        )
        if self._postgres_enabled and self._repo is not None:
            self._repo.record_run_start(
                refinery_run_id=refinery_run_id,
                source_mode=source_mode,
                source_run_id=source_run_id,
                requirements_count=requirements_count,
                settings_snapshot=settings_snapshot,
            )

    def on_run_end(
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
        log.info(
            "refinery_run_ended",
            refinery_run_id=refinery_run_id,
            converged=converged_count,
            short_circuited=short_circuited_count,
            aborted=aborted_count,
        )
        if self._postgres_enabled and self._repo is not None:
            self._repo.record_run_end(
                refinery_run_id=refinery_run_id,
                converged_count=converged_count,
                short_circuited_count=short_circuited_count,
                aborted_count=aborted_count,
                duration_seconds=duration_seconds,
                total_cost_usd=total_cost_usd,
                total_tokens_in=total_tokens_in,
                total_tokens_out=total_tokens_out,
            )

    # ── Per-debate lifecycle ──────────────────────────────────────────────

    def on_debate_end(
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
    ) -> int | None:
        """Write one row to ``refinery_debates`` and emit histogram/counter.
        Returns the debate_id so subsequent per-round / per-violation / per-
        research writes can reference it."""

        if self._prometheus_enabled:
            prom.observe_refinery_rounds(rounds=rounds_executed)
            prom.observe_refinery_convergence(outcome=convergence_status)
        if self._postgres_enabled and self._repo is not None:
            return self._repo.record_debate(
                refinery_run_id=refinery_run_id,
                requirement_id=requirement_id,
                convergence_status=convergence_status,
                rounds_executed=rounds_executed,
                escalation_level=escalation_level,
                research_calls_used=research_calls_used,
                final_overall_score=final_overall_score,
                final_dimension_scores=final_dimension_scores,
                disagreement_score_max=disagreement_score_max,
                duration_seconds=duration_seconds,
                cost_usd=cost_usd,
                termination_reason=termination_reason,
                reconcile_invoked=reconcile_invoked,
            )
        return None

    def on_round_end(
        self,
        *,
        debate_id: int | None,
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
        if self._prometheus_enabled:
            if disagreement_score is not None:
                prom.observe_refinery_disagreement(score=disagreement_score)
            for dim, score in (judge_dimensions or {}).items():
                prom.observe_refinery_judge_score(dimension=dim, score=score)
        if self._postgres_enabled and self._repo is not None and debate_id is not None:
            self._repo.record_round(
                debate_id=debate_id,
                round_number=round_number,
                critic_count=critic_count,
                critic_blockers_count=critic_blockers_count,
                critic_warnings_count=critic_warnings_count,
                rule_violations_count=rule_violations_count,
                rule_warnings_count=rule_warnings_count,
                judge_overall=judge_overall,
                judge_dimensions=judge_dimensions,
                disagreement_score=disagreement_score,
                router_decision=router_decision,
                duration_seconds=duration_seconds,
            )

    # ── LLM call (DUAL-WRITE) ────────────────────────────────────────────

    def on_llm_call(
        self,
        *,
        refinery_run_id: str,
        role: str,
        kind: str,
        model: str,
        requirement_id: str | None = None,
        round_number: int | None = None,
        reasoning_effort: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cache_read_tokens: int = 0,
        latency_ms: int | None = None,
        cost_usd: float = 0.0,
        error: str | None = None,
        tool_calls: list[dict] | None = None,
    ) -> None:
        if self._prometheus_enabled:
            prom.observe_refinery_role_call(role=role)
            prom.observe_refinery_cost(role=role, cost_usd=cost_usd)
            prom.observe_refinery_tokens(
                role=role, tokens_in=tokens_in, tokens_out=tokens_out,
            )
        if self._postgres_enabled and self._repo is not None:
            self._repo.record_llm_call(
                refinery_run_id=refinery_run_id,
                role=role,
                kind=kind,
                model=model,
                requirement_id=requirement_id,
                round_number=round_number,
                reasoning_effort=reasoning_effort,
                tokens_in=tokens_in or None,
                tokens_out=tokens_out or None,
                cache_read_tokens=cache_read_tokens or None,
                latency_ms=latency_ms,
                cost_usd=cost_usd if cost_usd else None,
                error=error,
                tool_calls=tool_calls,
            )

    # ── Rule violations, research sources, memory audits ─────────────────

    def on_rule_violation(
        self,
        *,
        debate_id: int | None,
        round_number: int,
        rule_id: str,
        severity: str,
        dimension: str,
        finding: str,
        suggested_fix: str | None = None,
        injected_as_critique: bool = False,
    ) -> None:
        if self._prometheus_enabled:
            prom.observe_refinery_rule_violation(
                rule_id=rule_id, severity=severity, dimension=dimension,
            )
        if self._postgres_enabled and self._repo is not None and debate_id is not None:
            self._repo.record_rule_violation(
                debate_id=debate_id,
                round_number=round_number,
                rule_id=rule_id,
                severity=severity,
                dimension=dimension,
                finding=finding,
                suggested_fix=suggested_fix,
                injected_as_critique=injected_as_critique,
            )

    def on_research_source(
        self,
        *,
        debate_id: int | None,
        round_number: int,
        tier: int | SourceTier,
        provider: str,
        url: str | None,
        title: str | None,
        propagated: bool,
        insight_confidence: float | None = None,
    ) -> None:
        tier_int = int(tier) if isinstance(tier, SourceTier) else int(tier)
        if self._prometheus_enabled:
            prom.observe_refinery_research_tier(tier=tier_int)
            if not propagated and tier_int == int(SourceTier.T5_WEB):
                prom.observe_refinery_t5_propagation_rejected()
        if self._postgres_enabled and self._repo is not None and debate_id is not None:
            self._repo.record_research_source(
                debate_id=debate_id,
                round_number=round_number,
                tier=tier_int,
                provider=provider,
                url=url,
                title=title,
                propagated=propagated,
                insight_confidence=insight_confidence,
            )

    def on_memory_audit(
        self,
        *,
        refinery_run_id: str,
        suggested_memory_id: str,
        kind: MemoryKind | str,
        source_role: str,
        validation_status: ValidationStatus | str,
        outcome: str,
        debate_id: int | None = None,
        existing_memory_id: str | None = None,
        similarity: float | None = None,
        provenance_source_tier_mix: list[int] | None = None,
        provenance_confidence: float | None = None,
    ) -> None:
        kind_s = kind.value if isinstance(kind, MemoryKind) else kind
        vs = validation_status.value if isinstance(validation_status, ValidationStatus) else validation_status
        if self._prometheus_enabled:
            prom.observe_refinery_memory_suggested(kind=kind_s, source_role=source_role)
            prom.observe_refinery_memory_audit(outcome=outcome)
            if outcome == "dedup_blocked":
                prom.observe_refinery_memory_dedup_blocked(kind=kind_s)
        if self._postgres_enabled and self._repo is not None:
            self._repo.record_memory_audit(
                refinery_run_id=refinery_run_id,
                suggested_memory_id=suggested_memory_id,
                kind=kind_s,
                source_role=source_role,
                validation_status=vs,
                outcome=outcome,
                debate_id=debate_id,
                existing_memory_id=existing_memory_id,
                similarity=similarity,
                provenance_source_tier_mix=provenance_source_tier_mix,
                provenance_confidence=provenance_confidence,
            )
