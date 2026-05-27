"""Router + escalation + reconcile + research wiring tests.

Covers the three terminal convergence outcomes (converged /
short_circuited / aborted), the termination invariants, and the
conditional edges that distinguish them.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    CritiqueDimension,
    EvaluationScore,
)
from dark_factory.api.refinery.debate.graph import (
    CRITIC_ROLES,
    _route_after_score,
    run_debate,
)
from langgraph.types import Send
from dark_factory.api.refinery.research import ResearchNote
from dark_factory.api.refinery.roles.judge import JudgeRole
from dark_factory.api.refinery.roles.product import ProductRole

from tests.refinery.conftest import (
    debate_kwargs,
    forced_judge_score,
    patched_product_propose,
)


# ─────────────────────────────────────────────────────────────────────
# Router truth-table — pure _route_after_score(state) tests
# ─────────────────────────────────────────────────────────────────────


def _state(
    *,
    round_number: int = 1,
    max_rounds: int = 3,
    score: dict | None = None,
    research_call_cap: int = 1,
    research_calls_used: int = 0,
    escalation_cap: int = 1,
    escalation_level: int = 0,
    aborted_reason: str | None = None,
) -> dict:
    state: dict = {
        "round_number": round_number,
        "max_rounds": max_rounds,
        "scores_by_round": {round_number: score} if score else {},
        "research_call_cap": research_call_cap,
        "research_calls_used": research_calls_used,
        "escalation_cap": escalation_cap,
        "escalation_level": escalation_level,
        "requirement": {"id": "req-1"},
    }
    if aborted_reason:
        state["aborted_reason"] = aborted_reason
    return state


# Truth-table parametrization: (scenario_id, _state kwargs, expected outcome)
_ROUTER_CASES = [
    ("abort_short_circuits",
     {"aborted_reason": "generator failed"},
     "finalize"),
    ("passed_finalizes",
     {"score": {"passed": True, "overall": 0.9}},
     "finalize"),
    ("max_rounds_without_convergence_reconciles",
     {"round_number": 3, "max_rounds": 3,
      "score": {"passed": False, "overall": 0.4}},
     "reconcile"),
    ("missing_external_info_triggers_research",
     {"score": {"passed": False, "overall": 0.5,
                "missing_external_info": True}},
     "research"),
    ("research_budget_exhausted_falls_through",
     {"score": {"passed": False, "overall": 0.5,
                "missing_external_info": True},
      "research_call_cap": 1, "research_calls_used": 1},
     "critic_fanout"),
    ("high_disagreement_escalates",
     {"score": {"passed": False, "overall": 0.5,
                "disagreement_score": 0.85}},
     "escalate"),
    ("escalation_exhausted_falls_through",
     {"score": {"passed": False, "overall": 0.5,
                "disagreement_score": 0.85},
      "escalation_cap": 1, "escalation_level": 1},
     "critic_fanout"),
    ("score_error_finalizes_gracefully",
     {"score": {"passed": False, "error": "judge 503"}},
     "finalize"),
    ("default_next_round",
     {"score": {"passed": False, "overall": 0.5}},
     "critic_fanout"),
]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [(c[1], c[2]) for c in _ROUTER_CASES],
    ids=[c[0] for c in _ROUTER_CASES],
)
def test_route_after_score_truth_table(kwargs, expected):
    result = _route_after_score(_state(**kwargs))
    if expected == "critic_fanout":
        # Non-terminal: router must emit one Send per critic role so
        # LangGraph re-dispatches the full panel in parallel. A plain
        # string here would route to the single ``critic`` node with no
        # ``_critic_role`` overlay and silently drop the panel down to
        # a placeholder.
        assert isinstance(result, list)
        assert len(result) == len(CRITIC_ROLES)
        assert all(isinstance(s, Send) and s.node == "critic" for s in result)
        assert {s.arg["_critic_role"] for s in result} == set(CRITIC_ROLES)
    else:
        assert result == expected


# ─────────────────────────────────────────────────────────────────────
# End-to-end debate tests using shared conftest fixtures
# ─────────────────────────────────────────────────────────────────────


def test_debate_converges_when_score_passes(
    fake_refined, pipeline_config, stub_critic_registry,
):
    with patched_product_propose(fake_refined), forced_judge_score(passing=True):
        terminal = run_debate(
            **debate_kwargs(
                max_rounds=2,
                refinery_run_id="refinery-test-converges",
            ),
            config=pipeline_config,
            registry=stub_critic_registry,
        )
    assert terminal["final_trace"]["convergence_status"] == ConvergenceStatus.CONVERGED.value


def test_debate_short_circuits_when_max_rounds_hit_without_convergence(
    fake_refined, pipeline_config, stub_critic_registry,
):
    with patched_product_propose(fake_refined), forced_judge_score(passing=False):
        terminal = run_debate(
            **debate_kwargs(
                max_rounds=2,
                research_call_cap=0,
                escalation_cap=0,
                strong_model="gpt-5.4-strong",
                refinery_run_id="refinery-test-short-circuit",
            ),
            config=pipeline_config,
            registry=stub_critic_registry,
        )
    assert terminal["final_trace"]["convergence_status"] == ConvergenceStatus.SHORT_CIRCUITED.value


def test_debate_redispatches_full_panel_on_non_convergent_score(
    fake_refined, pipeline_config, stub_critic_registry,
):
    """After a non-convergent score, the router must bounce back to the
    entire critic panel — not a single placeholder. Regression test for
    the routing fix where ``_route_after_score`` returning the string
    ``"next_round"`` silently dropped the round to one placeholder
    critique because the bare ``critic`` node received no
    ``_critic_role`` overlay."""

    with patched_product_propose(fake_refined), forced_judge_score(passing=False):
        terminal = run_debate(
            **debate_kwargs(
                max_rounds=2,
                research_call_cap=0,
                escalation_cap=0,
                strong_model="gpt-5.4-strong",
                refinery_run_id="refinery-test-fanout-round2",
            ),
            config=pipeline_config,
            registry=stub_critic_registry,
        )
    critiques_by_round = terminal["final_trace"]["critiques_by_round"]
    # Round 1 fan-out: from the generator edge (was already correct).
    round_1_roles = {c["author_role"] for c in critiques_by_round[1]}
    assert round_1_roles == set(CRITIC_ROLES), (
        f"round 1 missing critics: {set(CRITIC_ROLES) - round_1_roles}"
    )
    # Round 2 fan-out: from the score router. The bug this guards
    # against produced a single placeholder critique here.
    round_2_roles = {c["author_role"] for c in critiques_by_round[2]}
    assert round_2_roles == set(CRITIC_ROLES), (
        f"round 2 routing regressed — only saw critics: {round_2_roles}; "
        f"missing: {set(CRITIC_ROLES) - round_2_roles}"
    )


def test_debate_aborts_when_generator_crashes(
    pipeline_config, stub_critic_registry,
):
    def _boom(self, requirement, context):  # noqa: ARG001 — stub
        raise RuntimeError("generator outage")

    with patch.object(ProductRole, "propose", _boom):
        terminal = run_debate(
            **debate_kwargs(
                max_rounds=2,
                refinery_run_id="refinery-test-abort",
            ),
            config=pipeline_config,
            registry=stub_critic_registry,
        )
    trace = terminal["final_trace"]
    assert trace["convergence_status"] == ConvergenceStatus.ABORTED.value
    assert terminal["final_refined"] is None


def test_debate_research_node_uses_injected_factory(
    fake_refined, pipeline_config, stub_critic_registry,
):
    """Router branches to research when Judge flags missing_external_info.
    The injected factory returns a canned ResearchNote."""

    # Score once: missing_external_info=True → research.
    # Score twice: passed=True → finalize.
    score_calls = {"n": 0}

    def _alternating(self, draft, context, trace=None):  # noqa: ARG001
        score_calls["n"] += 1
        if score_calls["n"] == 1:
            return EvaluationScore(
                dimensions={d.value: 0.6 for d in CritiqueDimension},
                dimensions_semantic_raw={d.value: 0.6 for d in CritiqueDimension},
                reasons={}, overall=0.6, passed=False,
                missing_external_info=True, model_used="stub-missing",
            )
        return EvaluationScore(
            dimensions={d.value: 0.92 for d in CritiqueDimension},
            dimensions_semantic_raw={d.value: 0.92 for d in CritiqueDimension},
            reasons={}, overall=0.92, passed=True, model_used="stub-pass",
        )

    calls: list[str] = []

    class _FakeResearch:
        def run(self, *, query, round_number=0):
            calls.append(query)
            return ResearchNote(query=query)

    with patched_product_propose(fake_refined), patch.object(
        JudgeRole, "score", _alternating,
    ):
        terminal = run_debate(
            **debate_kwargs(
                max_rounds=3,
                escalation_cap=0,
                refinery_run_id="refinery-test-research",
            ),
            config=pipeline_config,
            registry=stub_critic_registry,
            research_agent_factory=lambda: _FakeResearch(),
        )
    assert calls, "expected research node to invoke the factory"
    trace = terminal["final_trace"]
    assert trace["research_calls_used"] >= 1
    assert trace["convergence_status"] == ConvergenceStatus.CONVERGED.value


def test_debate_escalate_node_bumps_model_and_level(
    fake_refined, pipeline_config, stub_critic_registry,
):
    """High disagreement on round 1 → escalate node switches base_model
    to strong_model, increments escalation_level, and re-runs critics."""

    score_calls = {"n": 0}

    def _escalate_then_pass(self, draft, context, trace=None):  # noqa: ARG001
        score_calls["n"] += 1
        if score_calls["n"] == 1:
            return EvaluationScore(
                dimensions={d.value: 0.55 for d in CritiqueDimension},
                dimensions_semantic_raw={d.value: 0.55 for d in CritiqueDimension},
                reasons={}, overall=0.55, passed=False,
                disagreement_score=0.85, model_used="stub-base",
            )
        return EvaluationScore(
            dimensions={d.value: 0.9 for d in CritiqueDimension},
            dimensions_semantic_raw={d.value: 0.9 for d in CritiqueDimension},
            reasons={}, overall=0.9, passed=True, model_used="stub-strong",
        )

    with patched_product_propose(fake_refined), patch.object(
        JudgeRole, "score", _escalate_then_pass,
    ):
        terminal = run_debate(
            **debate_kwargs(
                max_rounds=5,
                research_call_cap=0,
                escalation_cap=1,
                strong_model="gpt-5.4-strong",
                refinery_run_id="refinery-test-escalate",
            ),
            config=pipeline_config,
            registry=stub_critic_registry,
        )
    trace = terminal["final_trace"]
    assert trace["escalation_level"] == 1
    assert trace["convergence_status"] == ConvergenceStatus.CONVERGED.value
