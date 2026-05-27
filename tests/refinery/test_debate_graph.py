"""Debate graph + Judge synthesis + FallbackJudge tests.

Smoke-tests the ``generator → synthesize → score → finalize`` LangGraph
subgraph with a mocked legacy runner so no real LLM calls happen. Also
exercises JudgeRole.defend + JudgeRole.score + FallbackJudge scoring
heuristics.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    Critique,
    CritiqueDimension,
    Draft,
    DraftSpec,
    EvaluationScore,
    RoleContext,
    Severity,
)
from dark_factory.api.refinery.debate.graph import build_debate_graph, run_debate
from dark_factory.api.refinery.debate.runner import run_phase2_generator
from dark_factory.api.refinery.judge.fallback_judge import FallbackJudge
from dark_factory.api.refinery.models import RefinedRequirement
from dark_factory.api.refinery.roles.judge import JudgeRole
from dark_factory.api.refinery.roles.product import ProductRole
from dark_factory.api.refinery.roles.registry import RoleRegistry

from tests.refinery.conftest import (
    debate_kwargs,
    forced_judge_score,
    patched_product_propose,
)


_JUDGE_CTX = RoleContext(role="judge", requirement_id="req-1")


# ─────────────────────────────────────────────────────────────────────
# FallbackJudge — parametrised across strong/vague/short-circuit drafts
# ─────────────────────────────────────────────────────────────────────


def _strong_draft() -> Draft:
    return Draft(
        requirement_id="req-1",
        title="OAuth2 authorization-code flow",
        description=(
            "Given a valid client when authorize is called then a redirect occurs. "
            "Response time must be under 300ms at 100 req/s under 2 replicas."
        ),
        priority="high",
        tags=["auth", "security", "performance"],
        suggested_specs=[
            DraftSpec(
                title="OAuth2 flow", capability="auth", description="...",
                acceptance_criteria=[
                    "GIVEN a valid client WHEN authorize is called THEN a redirect occurs",
                    "GIVEN an invalid code WHEN token is called THEN 400 is returned",
                    "Token endpoint latency < 300ms at 100 req/s",
                ],
            )
        ],
        produced_by="product",
    )


def _vague_draft() -> Draft:
    return Draft(
        requirement_id="req-1", title="Fast system",
        description="The system should be fast, secure, robust, and scalable.",
        priority="high", produced_by="product",
    )


def _short_circuit_draft() -> Draft:
    return Draft(
        requirement_id="req-1", title="Unconverged auth feature",
        description="A " * 40, priority="high", produced_by="judge",
        convergence_status=ConvergenceStatus.SHORT_CIRCUITED,
        unresolved_points=["rate-limit strategy", "refresh-token lifetime"],
    )


def test_fallback_judge_scores_strong_draft_above_threshold():
    score = FallbackJudge().score(_strong_draft(), _JUDGE_CTX)
    assert isinstance(score, EvaluationScore)
    assert score.fallback_used is True
    assert set(score.dimensions) == {
        "clarity", "testability", "feasibility", "completeness", "risk_coverage",
        "reversibility",
    }
    assert all(v >= 0.7 for v in score.dimensions.values()), score.dimensions
    assert score.passed is True


@pytest.mark.parametrize(
    ("draft_factory", "low_dim"),
    [(_vague_draft, "clarity"), (_short_circuit_draft, "feasibility")],
    ids=["vague_draft_fails_clarity", "short_circuit_draft_fails_feasibility"],
)
def test_fallback_judge_flags_bad_drafts(draft_factory, low_dim):
    score = FallbackJudge().score(draft_factory(), _JUDGE_CTX)
    assert score.dimensions[low_dim] < 0.7
    assert score.passed is False


# ─────────────────────────────────────────────────────────────────────
# JudgeRole verbs
# ─────────────────────────────────────────────────────────────────────


def test_judge_defend_marks_converged_and_bumps_iteration():
    judge = JudgeRole()
    draft = Draft(
        requirement_id="req-1", title="T", description="D", priority="medium",
        produced_by="product", iteration=0,
    )
    crit = Critique(
        author_role="security", dimension=CritiqueDimension.RISK_COVERAGE,
        severity=Severity.BLOCKER, finding="missing threat model",
        proposed_fix="require STRIDE analysis",
    )
    rebuttal = judge.defend(draft, [crit], _JUDGE_CTX)
    assert rebuttal.revised_draft.iteration == 1
    assert rebuttal.revised_draft.convergence_status == ConvergenceStatus.CONVERGED
    assert rebuttal.revised_draft.produced_by == "judge"
    assert rebuttal.entries[0].action == "accepted"


def test_judge_defend_handles_empty_critiques():
    judge = JudgeRole()
    draft = Draft(
        requirement_id="req-1", title="T", description="D", priority="medium",
        produced_by="product",
    )
    rebuttal = judge.defend(draft, [], _JUDGE_CTX)
    assert rebuttal.revised_draft.iteration == 1
    assert rebuttal.entries == []


def test_judge_reconcile_unresolved_marks_short_circuit():
    judge = JudgeRole()
    draft = Draft(
        requirement_id="req-1", title="T", description="D", priority="medium",
        produced_by="product", iteration=0,
    )
    critiques = [
        Critique(
            author_role="engineering", dimension=CritiqueDimension.FEASIBILITY,
            severity=Severity.BLOCKER,
            finding="proposed stack doesn't exist in our fleet",
            proposed_fix="pick a vetted framework",
        ),
    ]
    rebuttal = judge.reconcile_unresolved(draft, critiques, [], [], _JUDGE_CTX)
    assert rebuttal.mode == "reconcile_unresolved"
    assert rebuttal.revised_draft.convergence_status == ConvergenceStatus.SHORT_CIRCUITED
    assert rebuttal.revised_draft.unresolved_points == [critiques[0].finding]
    assert rebuttal.revised_draft.open_questions == [critiques[0].proposed_fix]


# ─────────────────────────────────────────────────────────────────────
# Registry — JudgeRole auto-registered alongside Product
# ─────────────────────────────────────────────────────────────────────


def test_registry_auto_registers_judge_role(pipeline_config):
    registry = RoleRegistry(pipeline_config)
    assert registry.has("judge") is True
    assert isinstance(registry.get("judge"), JudgeRole)


def test_registry_enabled_roles_contains_product_and_judge(pipeline_config):
    registry = RoleRegistry(pipeline_config)
    assert {"product", "judge"}.issubset(set(registry.enabled_roles()))


# ─────────────────────────────────────────────────────────────────────
# Debate graph — end-to-end smoke with mocked legacy runner
# ─────────────────────────────────────────────────────────────────────


def test_debate_graph_runs_generator_synthesize_score_finalize(
    fake_refined, pipeline_config, stub_critic_registry,
):
    """End-to-end smoke: generator → critics → synthesize → score →
    finalize. Uses stubbed critics + forced-pass Judge.score so we
    don't require a live LLM; the happy path ends in CONVERGED on
    the first round."""

    with patched_product_propose(fake_refined), forced_judge_score(passing=True):
        terminal = run_debate(
            **debate_kwargs(
                title="Original", description="Original description",
                refinery_run_id="refinery-test-smoke",
            ),
            config=pipeline_config,
            registry=stub_critic_registry,
        )
    assert terminal["final_refined"] is not None
    trace = terminal["final_trace"]
    assert trace["convergence_status"] == ConvergenceStatus.CONVERGED.value
    assert trace["rounds_executed"] == 1
    assert len(trace["draft_history"]) == 2
    assert trace["draft_history"][0]["author_role"] == "product"
    assert trace["draft_history"][1]["author_role"] == "judge"
    assert trace["final_score"] is not None
    assert "dimensions" in trace["final_score"]


def test_debate_graph_aborts_cleanly_on_generator_failure(pipeline_config):
    """Generator raising causes the graph to walk to finalize via the
    plain edge; finalize sees no draft_history and emits an ABORTED
    trace."""

    def _boom(self, requirement, context):  # noqa: ARG001 — stub
        raise RuntimeError("model outage")

    with patch.object(ProductRole, "propose", _boom):
        terminal = run_debate(
            **debate_kwargs(refinery_run_id="refinery-test-abort"),
            config=pipeline_config,
        )

    trace = terminal["final_trace"]
    assert trace["convergence_status"] == ConvergenceStatus.ABORTED.value
    assert terminal["final_refined"] is None
    assert any("generator" in e.get("node", "") for e in trace["errors"])


def test_debate_graph_emits_progress_events_for_each_node(
    fake_refined, pipeline_config,
):
    emitted: list[dict] = []
    with patched_product_propose(fake_refined), forced_judge_score(passing=True):
        run_debate(
            **debate_kwargs(
                refinery_run_id="refinery-test-events", on_progress=emitted.append,
            ),
            config=pipeline_config,
        )
    events = {e.get("debate_event") for e in emitted if "debate_event" in e}
    expected = {"generator_started", "draft_ready", "synthesize_started",
                "synthesis_ready", "score_started", "score_ready", "finalized"}
    assert expected.issubset(events)


def test_build_debate_graph_returns_a_compiled_runnable(pipeline_config):
    graph = build_debate_graph(pipeline_config)
    assert callable(getattr(graph, "invoke", None))


# ─────────────────────────────────────────────────────────────────────
# Phase-2-style runner still works through the graph path
# ─────────────────────────────────────────────────────────────────────


def test_run_phase2_generator_returns_refined_requirement_shape(
    fake_refined, pipeline_config,
):
    with patched_product_propose(fake_refined), forced_judge_score(passing=True):
        refined, memories, _output = run_phase2_generator(
            req={"id": "req-1", "title": "Original", "description": "Original description",
                 "priority": "medium", "tags": []},
            all_requirements=[],
            run_context=None,
            tmpdir="/tmp/refinery-test",
            max_turns=5,
            timeout_seconds=60.0,
            model="gpt-5.4",
            reasoning_effort="xhigh",
            on_progress=None,
            config=pipeline_config,
            refinery_run_id="refinery-test-runner",
        )
    assert isinstance(refined, RefinedRequirement)
    assert refined.id == "req-1"
    assert refined.title == fake_refined.title
    assert memories == []
