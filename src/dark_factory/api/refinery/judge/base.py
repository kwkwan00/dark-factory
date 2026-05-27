"""Judge ABC — entry points for the per-req gate and Phase-3 cross-req review.

The production path is the combined three-phase pipeline (Phase A rules
→ Phase B LLM with rule context → Phase C rule→dimension penalties) which
lives in ``composition.py`` (added in Phase 7 of the rollout). This base
ABC defines only the interface so Phase 1 can stand up empty stubs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dark_factory.api.refinery.contracts import (
    CrossReviewReport,
    Draft,
    EvaluationScore,
    RoleContext,
)


class Judge(ABC):
    """Two entry points — per-requirement gating and cross-requirement review."""

    @abstractmethod
    def score(
        self,
        draft: Draft,
        context: RoleContext,
        trace: "DebateTrace | None" = None,  # type: ignore[name-defined]
    ) -> EvaluationScore:
        """Per-requirement gating. Used inside the debate subgraph."""

    @abstractmethod
    def review_set(
        self,
        refined_set: list[Draft],
        traces: list["DebateTrace"],  # type: ignore[name-defined]
        run_context: dict,
    ) -> CrossReviewReport:
        """Phase-3 cross-requirement adjudication (replaces the legacy
        single-pass reconciliation agent)."""
