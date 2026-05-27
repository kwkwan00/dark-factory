"""DeepEvalJudge — the production Phase-B semantic scorer.

The GEval metric builders below (one per ``CritiqueDimension``) mirror
the pattern at ``src/dark_factory/evaluation/metrics.py`` so the
Settings-tab model swap (``set_eval_model``) propagates here
automatically. Each metric's
criteria text references the rule findings that Phase A attached to
the ``RoleContext.evidence`` bag under the ``rule_findings`` key — the
LLM is prompt-instructed to agree with deterministic findings when
scoring its dimension.

The score() method catches all DeepEval exceptions and lets the
CombinedJudgePipeline's fallback kick in — callers never see a raw
provider error. The GEval imports are lazy-imported inside
``_build_metric()`` so environments without the DeepEval package can
still instantiate ``DeepEvalJudge()``.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from dark_factory.api.refinery.contracts import (
    CritiqueDimension,
    Draft,
    EvaluationScore,
    RoleContext,
)
from dark_factory.api.refinery.judge.base import Judge
from dark_factory.api.refinery.roles._shared.rule_findings import (
    render_rule_findings_block,
)
from dark_factory.api.refinery.roles._shared.stack import STACK_DESCRIPTION
from dark_factory.log import trace_methods

log = structlog.get_logger()

# The RULE_FINDINGS preamble gets appended to every GEval criteria text
# so the LLM scorer sees the deterministic rule output and is instructed
# to agree with it. Phase C still hard-clamps rule-blocker-affected
# dimensions at 0.5 as a backstop — this preamble makes the LLM agree
# with Phase C so the clamp rarely fires in practice.
_RULE_FINDINGS_PREAMBLE = """

Deterministic rule-based findings (ground truth — factor into your score):
{rule_findings}

When a BLOCKER finding is tagged to this dimension, your score must be
≤ 0.5 — the rule layer has deterministically verified a specific flaw
you cannot overrule. Explain in your reasoning which finding(s) drove
your score.

"""


_CRITERIA_TEXTS: dict[str, str] = {
    "clarity": (
        "Assess whether the refined requirement is unambiguous. It should "
        "contain exactly one interpretable intent, avoid vague modifiers "
        "(fast, secure, scalable, robust) without concrete measures, and "
        "use consistent terminology. Score low if multiple readers could "
        "plausibly disagree about what is in scope."
    ),
    "testability": (
        "Evaluate whether the requirement can be verified with concrete "
        "acceptance tests. Scenarios should present WHEN/THEN shapes, "
        "outputs must be observable, numbers must carry units, and every "
        "acceptance criterion should have a pass/fail oracle."
    ),
    "feasibility": (
        "Judge whether the requirement is implementable with the declared "
        f"stack ({STACK_DESCRIPTION}). Penalise invocations of systems not "
        "present unless the requirement explicitly notes a migration. "
        "Consider the turn budget implied by refinery_max_turns."
    ),
    "completeness": (
        "Evaluate whether the requirement covers acceptance criteria, edge "
        "cases, and non-functional concerns (performance, authorization, "
        "observability) the context implies. Missing any explicit "
        "non-functional facet the retrieved evidence hints at should "
        "lower the score."
    ),
    "risk_coverage": (
        "Assess whether known risks from context (prior Mistakes, failed "
        "evals, prior Incidents) are surfaced with mitigations. Score 1.0 "
        "only if every retrieved Mistake is either mentioned or explicitly "
        "dismissed with rationale."
    ),
    "reversibility": (
        "Assess how hard it would be to undo this requirement if it ships "
        "and turns out wrong. Score high (≥ 0.8) when rollback paths are "
        "explicit, schema/API changes are additive or feature-flagged, "
        "data migrations are reversible, and customer-visible commitments "
        "are bounded. Score low when the requirement bakes in irreversible "
        "changes (destructive migrations, public-API contracts without "
        "versioning, regulatory commitments) without naming the rollback "
        "strategy."
    ),
}


@trace_methods
class DeepEvalJudge(Judge):
    """Production semantic scorer. Uses DeepEval's GEval when available;
    degrades to a structured fallback LLM call on error (handled by the
    outer CombinedJudgePipeline). Per-dimension metric construction
    happens inside ``score`` inside a try/except so a failing metric
    (e.g. missing OPENAI_API_KEY) can't hang the pipeline."""

    def __init__(
        self,
        *,
        per_dim_thresholds: dict[str, float] | None = None,
    ) -> None:
        self._thresholds = per_dim_thresholds or {
            d.value: 0.7 for d in CritiqueDimension
        }

    def score(
        self,
        draft: Draft,
        context: RoleContext,
        trace: Any = None,
    ) -> EvaluationScore:
        started = time.monotonic()
        rule_findings_block = render_rule_findings_block(
            context.evidence.get("rule_findings") if context.evidence else None,
        )

        try:
            from deepeval.test_case import LLMTestCase
            from dark_factory.evaluation.metrics import get_eval_model
        except ImportError as exc:  # pragma: no cover — optional dep
            raise RuntimeError(
                f"DeepEval / evaluation-metrics import failed: {exc}"
            ) from exc

        model_name = get_eval_model()
        dims: dict[str, float] = {}
        reasons: dict[str, str] = {}

        test_case = _to_test_case(LLMTestCase, draft, context)
        preamble = _RULE_FINDINGS_PREAMBLE.format(rule_findings=rule_findings_block)
        # Build the eval LLM once per call — same instance reused for all
        # five metrics. Kept inside try/except so a GPTModel constructor
        # that probes auth synchronously can't hang the whole scorer.
        try:
            from dark_factory.evaluation.metrics import _build_eval_llm
            eval_llm = _build_eval_llm()
        except Exception as exc:
            log.warning("refinery_deepeval_llm_init_failed", error=str(exc))
            eval_llm = None
        for dim in CritiqueDimension:
            criteria = _CRITERIA_TEXTS[dim.value] + preamble
            try:
                if eval_llm is None:
                    raise RuntimeError("eval_llm unavailable")
                metric = self._build_metric(dim.value, criteria, eval_llm)
                metric.measure(test_case)
                dims[dim.value] = float(metric.score or 0.0)
                reasons[dim.value] = str(getattr(metric, "reason", "") or "")
            except Exception as exc:
                log.warning(
                    "refinery_deepeval_metric_raised",
                    dimension=dim.value, error=str(exc),
                )
                dims[dim.value] = 0.5
                reasons[dim.value] = f"metric error: {exc}"

        latency = time.monotonic() - started
        overall = min(dims.values()) if dims else 0.0
        return EvaluationScore(
            dimensions=dims,
            dimensions_semantic_raw=dict(dims),
            reasons=reasons,
            overall=overall,
            passed=all(
                dims.get(d.value, 0.0) >= self._thresholds.get(d.value, 0.7)
                for d in CritiqueDimension
            ),
            aggregation="min",
            thresholds=dict(self._thresholds),
            model_used=model_name,
            latency_seconds=latency,
        )

    def review_set(
        self,
        refined_set: list[Draft],
        traces: list,
        run_context: dict,
    ) -> "CrossReviewReport":  # type: ignore[name-defined]
        """Cross-requirement review — pass-through stub.

        Structural integration is wired (the cross-review patcher
        applies ``CrossReviewReport`` mutations); the LLM-backed set
        review is intentionally left for a future follow-up.
        """

        from dark_factory.api.refinery.contracts import CrossReviewReport

        return CrossReviewReport(
            summary="DeepEvalJudge.review_set: pass-through stub."
        )

    # ── Internals ─────────────────────────────────────────────────────

    def _build_metric(self, dim_name: str, criteria: str, eval_llm: Any):
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCaseParams

        return GEval(
            name=f"Refinery — {dim_name}",
            criteria=criteria,
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.INPUT,
            ],
            threshold=self._thresholds.get(dim_name, 0.7),
            model=eval_llm,
        )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _to_test_case(LLMTestCase, draft: Draft, context: RoleContext):
    """Build a DeepEval LLMTestCase from our Draft + RoleContext."""

    actual = json.dumps({
        "title": draft.title,
        "description": draft.description,
        "priority": draft.priority,
        "tags": list(draft.tags),
        "suggested_specs": [
            {"title": s.title, "capability": s.capability,
             "description": s.description,
             "acceptance_criteria": list(s.acceptance_criteria)}
            for s in draft.suggested_specs
        ],
        "relationships": [
            {"target_id": r.target_id, "type": r.type, "rationale": r.rationale}
            for r in draft.relationships
        ],
        "explicit_tradeoffs": list(draft.explicit_tradeoffs),
    }, indent=2)
    input_text = (
        f"Role: {context.role}\n"
        f"Requirement ID: {context.requirement_id}\n"
        f"Narrative evidence:\n{context.narrative or '(none)'}"
    )
    return LLMTestCase(input=input_text, actual_output=actual)
