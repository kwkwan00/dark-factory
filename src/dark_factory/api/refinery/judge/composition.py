"""The combined evaluation pipeline — fuses rules (Phase A) with the
LLM-as-judge (Phase B) and applies Phase C rule→dimension penalties to
produce a single unified ``EvaluationScore``.

Contract: router reads only ``EvaluationScore.passed`` and the signals
(``missing_external_info``, ``disagreement_score``). Everything else —
per-dimension scores, rule violations, which rules caused which penalty
— is preserved for the trace.
"""

from __future__ import annotations

from typing import Any

import structlog

from dark_factory.api.refinery.contracts import (
    CritiqueDimension,
    Draft,
    EvaluationScore,
    RoleContext,
    Severity,
)
from dark_factory.api.refinery.judge.base import Judge
from dark_factory.api.refinery.judge.rules_judge import RuleResult, RulesJudge
from dark_factory.log import trace_methods
from dark_factory.metrics.prometheus import (
    observe_refinery_eval_short_circuit,
    observe_refinery_rule_penalty_applied,
)

log = structlog.get_logger()


@trace_methods
class CombinedJudgePipeline:
    """Three-phase Judge composed over (rules_judge, semantic_judge).

    Usage pattern:

        pipeline = CombinedJudgePipeline(
            rules_judge=RulesJudge(disabled=cfg.refinery_rules_disabled),
            semantic_judge=DeepEvalJudge(),
            short_circuit_threshold=cfg.refinery_rules_short_circuit_threshold,
            short_circuit_enabled=cfg.refinery_rules_short_circuit_llm,
            blocker_cap=cfg.refinery_rules_penalty_blocker_cap,
            warning_delta=cfg.refinery_rules_penalty_warning_delta,
            overall_threshold=cfg.refinery_judge_overall_threshold,
            per_dim_thresholds=cfg.refinery_judge_thresholds,
        )
        score = pipeline.score(draft, context)

    The semantic_judge protocol is the ``Judge`` ABC — any class with
    ``.score(draft, context, trace)`` fits. DeepEvalJudge is the
    production choice; FallbackJudge is the degraded-mode fallback.
    """

    def __init__(
        self,
        *,
        rules_judge: RulesJudge,
        semantic_judge: Judge,
        fallback_judge: Judge | None = None,
        short_circuit_enabled: bool = False,
        short_circuit_threshold: int = 3,
        blocker_cap: float = 0.5,
        warning_delta: float = 0.1,
        overall_threshold: float = 0.8,
        per_dim_thresholds: dict[str, float] | None = None,
        aggregation: str = "min",
        rules_enabled: bool = True,
    ) -> None:
        self._rules = rules_judge
        self._semantic = semantic_judge
        self._fallback = fallback_judge
        self._short_circuit_enabled = short_circuit_enabled
        self._short_circuit_threshold = max(1, short_circuit_threshold)
        self._blocker_cap = blocker_cap
        self._warning_delta = warning_delta
        self._overall_threshold = overall_threshold
        self._per_dim_thresholds = per_dim_thresholds or {
            d.value: 0.7 for d in CritiqueDimension
        }
        self._aggregation = aggregation
        self._rules_enabled = rules_enabled

    # ── Public entry point ────────────────────────────────────────────

    def score(
        self,
        draft: Draft,
        context: RoleContext,
        trace: Any = None,
    ) -> EvaluationScore:
        # Phase A — rules first
        if self._rules_enabled:
            rule_result = self._rules.validate(draft, context, trace)
        else:
            rule_result = RuleResult()

        # Apply BEFORE Phase B so the LLM prompt, the clamp, the
        # synthetic-critique injection, and the trace all see the same
        # demoted shape — split paths would let the LLM and the trace
        # disagree about what the rules said.
        severity_overrides = (context.evidence or {}).get(
            "rule_severity_overrides"
        )
        if severity_overrides:
            from dark_factory.api.refinery.rule_severity import (
                apply_severity_overrides,
            )
            rule_result = apply_severity_overrides(rule_result, severity_overrides)

        # Short-circuit the LLM call on catastrophic rule failure
        llm_short_circuited = False
        if (
            self._short_circuit_enabled
            and len(rule_result.blockers) >= self._short_circuit_threshold
        ):
            semantic = self._stub_dimensions_from_rules(rule_result)
            llm_short_circuited = True
            observe_refinery_eval_short_circuit()
        else:
            # Phase B — LLM judge, with rule findings injected
            semantic = self._run_semantic(draft, context, trace, rule_result)

        # Phase C — rule→dimension penalties (deterministic backstop)
        final_dims = self._apply_rule_penalties(semantic.dimensions, rule_result)
        overall = self._aggregate(final_dims)

        # Confidence-calibration nudge: scale the configured threshold
        # by the multiplier the orchestrator stamped onto the evidence
        # bag from cross-run operator feedback. Identity (1.0) means no
        # signal yet; bounded to [0.85, 1.15] upstream so the nudge is
        # a tie-breaker, not a license to silently move the goalpost.
        calibration_mult = float(
            (context.evidence or {}).get("judge_threshold_multiplier") or 1.0
        )
        effective_threshold = self._overall_threshold * calibration_mult

        passed = (
            overall >= effective_threshold
            and all(
                final_dims.get(d.value, 0.0)
                >= self._per_dim_thresholds.get(d.value, 0.7)
                for d in CritiqueDimension
            )
            and not rule_result.blockers
        )

        # Post-penalty dim scores are observed by ObservabilityHub.on_round_end
        # downstream. Pre-penalty raw LLM scores aren't emitted from
        # elsewhere, so mirror them here for pre/post comparison in
        # Grafana. Skip if the caller disabled Prometheus entirely.
        from dark_factory.metrics.prometheus import observe_refinery_judge_score
        for dim_name, dim_score in semantic.dimensions.items():
            observe_refinery_judge_score(
                dimension=dim_name, score=dim_score, raw_llm=True,
            )

        return EvaluationScore(
            dimensions=final_dims,
            dimensions_semantic_raw=dict(semantic.dimensions),
            reasons=dict(semantic.reasons),
            overall=overall,
            passed=passed,
            aggregation=self._aggregation,
            thresholds=dict(self._per_dim_thresholds),
            overall_threshold=effective_threshold,
            model_used=semantic.model_used,
            latency_seconds=semantic.latency_seconds,
            missing_external_info=semantic.missing_external_info,
            disagreement_score=semantic.disagreement_score,
            rule_violations=rule_result.blockers,
            rule_warnings=rule_result.warnings,
            rules_engine_failed=rule_result.engine_failed,
            rules_engine_disabled=not self._rules_enabled,
            rules_skipped=list(rule_result.rules_skipped),
            llm_short_circuited=llm_short_circuited,
            fallback_used=semantic.fallback_used,
        )

    # ── Internals ─────────────────────────────────────────────────────

    def _run_semantic(
        self,
        draft: Draft,
        context: RoleContext,
        trace: Any,
        rule_result: RuleResult,
    ) -> EvaluationScore:
        """Run Phase B. On exception, fall back to the heuristic judge."""

        # Rule findings are attached to the evidence bag (RoleContext is
        # frozen, so we thread them via context.evidence['rule_findings']
        # — the DeepEvalJudge reads from that key to populate its prompt).
        try:
            enriched_evidence = dict(context.evidence)
            enriched_evidence["rule_findings"] = [
                v.model_dump() for v in rule_result.violations
            ]
            enriched_ctx = context.model_copy(
                update={"evidence": enriched_evidence},
            )
            return self._semantic.score(draft, enriched_ctx, trace)
        except Exception as exc:
            log.warning(
                "refinery_semantic_judge_raised",
                error=str(exc), falling_back=bool(self._fallback),
            )
            if self._fallback is None:
                # No fallback — return a degraded-but-non-crashing score
                return EvaluationScore(
                    dimensions={d.value: 0.5 for d in CritiqueDimension},
                    dimensions_semantic_raw={d.value: 0.5 for d in CritiqueDimension},
                    reasons={d.value: "semantic judge error"
                             for d in CritiqueDimension},
                    overall=0.5,
                    passed=False,
                    model_used="error",
                    fallback_used=True,
                )
            return self._fallback.score(draft, context, trace)

    def _stub_dimensions_from_rules(
        self, rule_result: RuleResult,
    ) -> EvaluationScore:
        """When we skip Phase B on catastrophic rule failure, produce a
        stub EvaluationScore whose dimensions reflect only what the
        rules tell us. Blocker-affected dimensions → 0.0."""

        dims = {d.value: 0.5 for d in CritiqueDimension}
        for v in rule_result.blockers:
            dims[v.dimension.value] = 0.0
        return EvaluationScore(
            dimensions=dims,
            dimensions_semantic_raw=dict(dims),
            reasons={v.dimension.value: v.finding
                     for v in rule_result.blockers},
            overall=min(dims.values()),
            passed=False,
            model_used="short-circuited",
        )

    def _apply_rule_penalties(
        self,
        semantic_dims: dict[str, float],
        rule_result: RuleResult,
    ) -> dict[str, float]:
        """Phase C — hard cap on blocker-affected dimensions, soft
        subtract on warning-affected dimensions."""

        final = {d.value: semantic_dims.get(d.value, 0.0)
                 for d in CritiqueDimension}
        for v in rule_result.violations:
            dim = v.dimension.value
            if v.severity == Severity.BLOCKER:
                if final.get(dim, 0.0) > self._blocker_cap:
                    final[dim] = self._blocker_cap
                observe_refinery_rule_penalty_applied(
                    dimension=dim, severity="blocker",
                )
            elif v.severity == Severity.WARNING:
                final[dim] = max(
                    0.0, final.get(dim, 0.0) - self._warning_delta,
                )
                observe_refinery_rule_penalty_applied(
                    dimension=dim, severity="warning",
                )
        return final

    def _aggregate(self, dims: dict[str, float]) -> float:
        if not dims:
            return 0.0
        if self._aggregation == "mean":
            return sum(dims.values()) / len(dims)
        # Default: minimum over dimensions (every dim must clear the bar)
        return min(dims.values())
