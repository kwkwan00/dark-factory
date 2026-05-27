"""Tests for the multi-pass set-level Judge (run_set_review)."""

from __future__ import annotations

from unittest.mock import patch

from dark_factory.api.refinery.contracts import (
    CrossReviewReport,
    Critique,
    CritiqueDimension,
    Draft,
    Severity,
)
from dark_factory.api.refinery.roles.judge import JudgeRole
from dark_factory.api.refinery.roles.registry import RoleRegistry
from dark_factory.api.refinery.set_review import run_set_review
from dark_factory.config import PipelineConfig


def _draft(req_id: str) -> Draft:
    return Draft(
        requirement_id=req_id,
        title=f"Title {req_id}",
        description=f"Description {req_id}",
        priority="medium",
        tags=[],
        suggested_specs=[],
        relationships=[],
        produced_by="test",
        iteration=0,
    )


def test_max_rounds_one_is_single_shot():
    """``max_rounds=1`` collapses to a single Judge.review_set call —
    the legacy behaviour."""

    review_calls: list[int] = []

    def _spy(self, refined_set, traces, run_context):
        review_calls.append(len(refined_set))
        return CrossReviewReport(summary="round-0 only")

    registry = RoleRegistry(PipelineConfig())

    with patch.object(JudgeRole, "review_set", _spy):
        report = run_set_review(
            refined_set=[_draft("r-1"), _draft("r-2")],
            run_context={"run_id": "run-1"},
            registry=registry,
            max_rounds=1,
        )

    assert report.summary == "round-0 only"
    assert review_calls == [2]


def test_no_blockers_after_round_zero_short_circuits():
    """When the first-pass report draws no critic blockers, the loop
    exits without re-invoking review_set even if max_rounds is high."""

    review_count = {"n": 0}

    def _spy(self, refined_set, traces, run_context):
        review_count["n"] += 1
        return CrossReviewReport(summary="initial draft")

    def _info_critique(self, draft, context):
        return Critique(
            author_role=getattr(self, "role_name", "engineering"),
            severity=Severity.INFO,
            dimension=CritiqueDimension.FEASIBILITY,
            finding="no concern after deliberate search of set scope",
            proposed_fix="",
        )

    from dark_factory.api.refinery.roles._shared.critic_role import (
        _CriticBaseRole,
    )

    registry = RoleRegistry(PipelineConfig())

    with patch.object(JudgeRole, "review_set", _spy), \
         patch.object(_CriticBaseRole, "critique", _info_critique):
        report = run_set_review(
            refined_set=[_draft("r-1"), _draft("r-2")],
            run_context={"run_id": "run-1"},
            registry=registry,
            max_rounds=3,
        )

    assert report.summary == "initial draft"
    # No round-1 rebut call because no critic returned a blocker.
    assert review_count["n"] == 1


def test_blocker_triggers_rebut_pass():
    """A critic blocker on round 1 causes the Judge to be re-invoked
    with the critiques carried in ``traces``."""

    review_calls: list[dict] = []

    def _spy(self, refined_set, traces, run_context):
        review_calls.append({"round": len(review_calls), "traces": list(traces)})
        return CrossReviewReport(summary=f"draft round {len(review_calls) - 1}")

    def _blocker(self, draft, context):
        return Critique(
            author_role=getattr(self, "role_name", "security"),
            severity=Severity.BLOCKER,
            dimension=CritiqueDimension.RISK_COVERAGE,
            finding="set-level review missed a recurring risk pattern across requirements",
            proposed_fix="reframe risk_areas to highlight the cross-req pattern",
        )

    from dark_factory.api.refinery.roles._shared.critic_role import (
        _CriticBaseRole,
    )

    registry = RoleRegistry(PipelineConfig())

    with patch.object(JudgeRole, "review_set", _spy), \
         patch.object(_CriticBaseRole, "critique", _blocker):
        report = run_set_review(
            refined_set=[_draft("r-1"), _draft("r-2")],
            run_context={"run_id": "run-1"},
            registry=registry,
            max_rounds=2,
        )

    # Two Judge.review_set calls — round 0 (initial) + round 1 (rebut).
    assert len(review_calls) == 2
    # Round 1 carried the critic findings as traces.
    assert review_calls[1]["traces"]  # non-empty
    assert review_calls[1]["traces"][0]["severity"] == "blocker"
    assert "round 1" in report.summary


def test_no_judge_returns_empty_report():
    """Missing judge in registry → empty CrossReviewReport, no
    crashes."""

    class _RegistryWithoutJudge:
        def has(self, name: str) -> bool:
            return False

        def get(self, name):
            raise KeyError(name)

    report = run_set_review(
        refined_set=[_draft("r-1"), _draft("r-2")],
        run_context=None,
        registry=_RegistryWithoutJudge(),
        max_rounds=3,
    )
    assert isinstance(report, CrossReviewReport)
    assert report.summary == ""
