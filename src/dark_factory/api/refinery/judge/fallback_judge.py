"""Fallback judge — deterministic heuristic scorer.

Two responsibilities:

1. Kick in automatically when DeepEval raises (import failure, provider
   503, metric timeout) so a model outage can't stall the refinery.
2. Offer a zero-LLM path for local/test mode.

The heuristic is intentionally simple — it looks at structural
properties of the draft (description length, acceptance-criteria count,
explicit tradeoffs, unresolved points) and maps them onto every
``CritiqueDimension``. ``DeepEvalJudge`` is the production scorer; this
module stays as the backstop.
"""

from __future__ import annotations

import re
from typing import Any

from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    CritiqueDimension,
    Draft,
    EvaluationScore,
    RoleContext,
)
from dark_factory.api.refinery.judge._lexicon import (
    GIVEN_WHEN_RE,
    IRREVERSIBLE_SIGNALS,
    MEASURE_RE,
    ROLLBACK_SIGNALS,
    VAGUE_WORDS,
    WHEN_THEN_RE,
)
from dark_factory.api.refinery.judge.base import Judge
from dark_factory.log import trace_methods

# Reuse the shared lexicon so fallback and rules can't drift.
_VAGUE_TOKENS = frozenset(VAGUE_WORDS)
_MEASURABLE_PATTERNS = (WHEN_THEN_RE, GIVEN_WHEN_RE, MEASURE_RE)


@trace_methods
class FallbackJudge(Judge):
    """Zero-LLM heuristic scorer used as the last-resort fallback."""

    def __init__(self, *, thresholds: dict[str, float] | None = None,
                 overall_threshold: float = 0.8) -> None:
        self._thresholds = thresholds or {
            "clarity": 0.7, "testability": 0.7, "feasibility": 0.7,
            "completeness": 0.7, "risk_coverage": 0.7,
            "reversibility": 0.7,
        }
        self._overall_threshold = overall_threshold

    def score(
        self,
        draft: Draft,
        context: RoleContext,
        trace: Any = None,
    ) -> EvaluationScore:
        dims = {
            "clarity": self._score_clarity(draft),
            "testability": self._score_testability(draft),
            "feasibility": self._score_feasibility(draft),
            "completeness": self._score_completeness(draft),
            "risk_coverage": self._score_risk_coverage(draft),
            "reversibility": self._score_reversibility(draft),
        }
        overall = min(dims.values())  # "min" aggregation by default
        passed = overall >= self._overall_threshold and all(
            dims[d] >= self._thresholds.get(d, 0.7) for d in dims
        )
        return EvaluationScore(
            dimensions=dims,
            dimensions_semantic_raw=dict(dims),
            reasons={d: "fallback heuristic" for d in dims},
            overall=overall,
            passed=passed,
            aggregation="min",
            thresholds=dict(self._thresholds),
            overall_threshold=self._overall_threshold,
            model_used="fallback:heuristic",
            fallback_used=True,
        )

    def review_set(self, refined_set, traces, run_context):  # type: ignore[override]
        """Cross-req review isn't covered by the fallback — the real
        DeepEvalJudge implements ``review_set``. Returning an empty
        report here keeps the debate subgraph runnable in smoke tests
        without forcing a DeepEval call."""

        from dark_factory.api.refinery.contracts import CrossReviewReport

        return CrossReviewReport(
            summary="FallbackJudge: cross-req review not implemented; pass-through."
        )

    # ── Per-dimension heuristics ──────────────────────────────────────

    def _score_clarity(self, d: Draft) -> float:
        """Clarity = (1 - vague_token_share) with a floor of 0.4 so a
        draft that mentions "secure" once but is otherwise specific
        doesn't crash to zero. Short drafts (<50 chars) cap at 0.5."""

        text = f"{d.title} {d.description}".lower()
        if len(text) < 50:
            return 0.5
        words = re.findall(r"\b[a-z-]+\b", text)
        if not words:
            return 0.5
        vague = sum(1 for w in words if w in _VAGUE_TOKENS)
        share = vague / len(words)
        return max(0.4, min(1.0, 1.0 - share * 4))

    def _score_testability(self, d: Draft) -> float:
        """Testability = fraction of acceptance criteria that match a
        measurable pattern. No suggested_specs → 0.5 by default."""

        specs = d.suggested_specs
        if not specs:
            return 0.5
        all_criteria = [c for s in specs for c in s.acceptance_criteria]
        if not all_criteria:
            return 0.3
        matched = sum(
            1 for c in all_criteria
            if any(p.search(c) for p in _MEASURABLE_PATTERNS)
        )
        return min(1.0, 0.2 + 0.8 * (matched / len(all_criteria)))

    def _score_feasibility(self, d: Draft) -> float:
        """Feasibility = high unless the draft carries a short-circuited
        convergence status, which signals the panel couldn't
        agree on an implementable path."""

        if d.convergence_status == ConvergenceStatus.SHORT_CIRCUITED:
            return 0.45
        if d.convergence_status == ConvergenceStatus.ABORTED:
            return 0.2
        # Priority mismatch with tag signals lowers feasibility slightly.
        return 0.8

    def _score_completeness(self, d: Draft) -> float:
        """Completeness = spec count × coverage, capped at 1.0."""

        if not d.suggested_specs:
            return 0.35
        # Require at least 3 criteria per spec for full credit.
        per_spec = [min(1.0, len(s.acceptance_criteria) / 3.0)
                    for s in d.suggested_specs]
        return min(1.0, 0.4 + 0.6 * (sum(per_spec) / len(per_spec)))

    def _score_risk_coverage(self, d: Draft) -> float:
        """Risk coverage = 0.5 + boost for each tag that signals a risk
        area is acknowledged (security, performance, compliance, etc.)."""

        risk_tags = {"security", "auth", "privacy", "compliance",
                     "performance", "scalability", "observability"}
        matched = sum(1 for t in d.tags if t.lower() in risk_tags)
        base = 0.5 + min(0.4, matched * 0.15)
        if d.unresolved_points:
            # Unresolved short-circuit points drag coverage down.
            base *= 0.7
        return min(1.0, base)

    def _score_reversibility(self, d: Draft) -> float:
        """Default 0.8 (presumed reasonably reversible when no destructive
        signal is present). Drafts must explicitly mention irreversible
        vocabulary OR be priority=critical without any rollback affordance
        to score below the per-dim threshold."""

        text = f"{d.title} {d.description}".lower()
        irreversible_hits = sum(1 for s in IRREVERSIBLE_SIGNALS if s in text)
        rollback_hits = sum(1 for s in ROLLBACK_SIGNALS if s in text)
        base = 0.8 - 0.2 * irreversible_hits + 0.1 * rollback_hits
        # priority=critical without any rollback strategy is the worst
        # case for blast radius — apply an extra penalty.
        if d.priority == "critical" and rollback_hits == 0:
            base -= 0.1
        return max(0.0, min(1.0, base))
