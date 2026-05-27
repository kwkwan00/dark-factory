"""Adversarial critics + fan-out + anti-rubber-stamp tests.

Covers:
- Rubber-stamp validator rejects vacuous / approval-style critiques.
- Retry loop produces a substantive critique on the second attempt,
  or a placeholder after two failures.
- Parallel fan-out via Send: all four critics run per round and their
  outputs merge into critiques_by_round.
- One critic crashing doesn't abort the round — the other three still
  produce critiques, the crashed one becomes a placeholder.
- Anti-collusion: all four critics rubber-stamp → all four become
  placeholders, round continues.
"""

from __future__ import annotations

import json

import pytest

from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    Critique,
    CritiqueDimension,
    Draft,
    RoleContext,
    Severity,
)
from dark_factory.api.refinery.debate.graph import CRITIC_ROLES, run_debate
from dark_factory.api.refinery.roles._shared.critic_base import (
    RubberStampRejected,
    call_critic_with_retry,
    is_rubber_stamp,
    parse_critique_json,
)
from dark_factory.api.refinery.roles.cost import CostRole
from dark_factory.api.refinery.roles.engineering import EngineeringRole
from dark_factory.api.refinery.roles.operations import OperationsRole
from dark_factory.api.refinery.roles.registry import RoleRegistry
from dark_factory.api.refinery.roles.product import ProductRole
from dark_factory.api.refinery.roles.security import SecurityRole

from tests.refinery.conftest import (
    debate_kwargs,
    forced_judge_score,
    patched_product_propose,
    substantive_critique_json,
)


# ─────────────────────────────────────────────────────────────────────
# Rubber-stamp validator — parametrised to replace 6 near-identical tests
# ─────────────────────────────────────────────────────────────────────


def _critique(
    *,
    finding: str = "",
    proposed_fix: str = "",
    severity: Severity = Severity.WARNING,
    dimension: CritiqueDimension = CritiqueDimension.FEASIBILITY,
) -> Critique:
    return Critique(
        author_role="engineering", severity=severity, dimension=dimension,
        finding=finding, proposed_fix=proposed_fix,
    )


_RUBBER_STAMP_CASES = [
    ("empty_finding",
     {"finding": "", "proposed_fix": "do something"}, True),
    ("short_finding",
     {"finding": "too short", "proposed_fix": "do something concrete"}, True),
    ("short_proposed_fix",
     {"finding": "this is a twenty-char substantive finding here",
      "proposed_fix": "short"}, True),
    ("approval_phrase_lgtm",
     {"finding": "LGTM from my perspective",
      "proposed_fix": "no changes needed"}, True),
    ("approval_phrase_looks_good",
     {"finding": "Looks good to me overall",
      "proposed_fix": "no changes needed"}, True),
    ("approval_phrase_no_concerns",
     {"finding": "No concerns worth flagging",
      "proposed_fix": "no changes needed"}, True),
    ("approval_phrase_nothing_to_add",
     {"finding": "Nothing to add here folks",
      "proposed_fix": "no changes needed"}, True),
    ("substantive",
     {"finding": "Token store backend is unspecified; affects scalability",
      "proposed_fix": "Pick Redis with TTL-based eviction"}, False),
    ("info_with_narrative",
     {"finding": (
        "Searched for OAuth2 threat-model gaps: checked token theft, "
        "PKCE, redirect-URI validation — all explicit in the draft."
      ),
      "proposed_fix": "",
      "severity": Severity.INFO,
      "dimension": CritiqueDimension.RISK_COVERAGE}, False),
    ("empty_info",
     {"finding": "", "proposed_fix": "",
      "severity": Severity.INFO,
      "dimension": CritiqueDimension.RISK_COVERAGE}, True),
]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [(c[1], c[2]) for c in _RUBBER_STAMP_CASES],
    ids=[c[0] for c in _RUBBER_STAMP_CASES],
)
def test_is_rubber_stamp(kwargs, expected):
    assert is_rubber_stamp(_critique(**kwargs)) is expected


# ─────────────────────────────────────────────────────────────────────
# parse_critique_json
# ─────────────────────────────────────────────────────────────────────


def test_parse_critique_json_rewrites_author_role():
    raw = json.dumps({
        "author_role": "SOMEBODY_ELSE",
        "dimension": "feasibility",
        "severity": "warning",
        "finding": "The proposed API shape conflicts with our existing contract",
        "proposed_fix": "align with the canonical schema",
    })
    c = parse_critique_json(
        raw, role_name="engineering",
        default_dimension=CritiqueDimension.FEASIBILITY,
    )
    assert c.author_role == "engineering"


def test_parse_critique_json_tolerates_fenced_output():
    raw = f"```json\n{substantive_critique_json()}\n```\nsome trailing prose"
    c = parse_critique_json(
        raw, role_name="engineering",
        default_dimension=CritiqueDimension.FEASIBILITY,
    )
    assert c.severity == Severity.WARNING


def test_parse_critique_json_raises_rubber_stamp_on_vacuous():
    raw = json.dumps({
        "author_role": "engineering",
        "severity": "warning",
        "finding": "Looks good",
        "proposed_fix": "",
    })
    with pytest.raises(RubberStampRejected):
        parse_critique_json(
            raw, role_name="engineering",
            default_dimension=CritiqueDimension.FEASIBILITY,
        )


def test_parse_critique_json_raises_value_error_on_bad_json():
    with pytest.raises(ValueError):
        parse_critique_json(
            "not json at all",
            role_name="engineering",
            default_dimension=CritiqueDimension.FEASIBILITY,
        )


# ─────────────────────────────────────────────────────────────────────
# call_critic_with_retry — the adversarial retry loop
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def empty_draft() -> Draft:
    return Draft(
        requirement_id="req-1", title="T", description="D",
        priority="medium", produced_by="product",
    )


@pytest.fixture
def empty_ctx() -> RoleContext:
    return RoleContext(role="engineering", requirement_id="req-1")


def _run_retry(generate, draft, ctx):
    return call_critic_with_retry(
        role_name="engineering",
        default_dimension=CritiqueDimension.FEASIBILITY,
        draft=draft, context=ctx, generate=generate, max_retries=1,
    )


def test_retry_loop_accepts_first_substantive_critique(empty_draft, empty_ctx):
    calls: list[str] = []

    def generate(d, c, followup):
        calls.append(followup)
        return substantive_critique_json()

    critique = _run_retry(generate, empty_draft, empty_ctx)
    assert critique.severity == Severity.WARNING
    assert len(calls) == 1
    assert calls[0] == ""


def test_retry_loop_retries_rubber_stamp_then_accepts(empty_draft, empty_ctx):
    attempts = 0

    def generate(d, c, followup):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return json.dumps({
                "author_role": "engineering", "severity": "warning",
                "finding": "Looks good", "proposed_fix": "",
            })
        return substantive_critique_json()

    critique = _run_retry(generate, empty_draft, empty_ctx)
    assert critique.severity == Severity.WARNING
    assert attempts == 2


def test_retry_loop_produces_placeholder_after_two_rubber_stamps(empty_draft, empty_ctx):
    def generate(d, c, followup):
        return json.dumps({
            "author_role": "engineering", "severity": "warning",
            "finding": "LGTM", "proposed_fix": "",
        })

    critique = _run_retry(generate, empty_draft, empty_ctx)
    assert critique.severity == Severity.INFO
    assert critique.confidence == 0.0
    err = (critique.error or "").lower()
    assert "rubber-stamp" in err or "rubber_stamp" in err


def test_retry_loop_recovers_from_transient_generator_exception(empty_draft, empty_ctx):
    attempts = 0

    def generate(d, c, followup):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient 503")
        return substantive_critique_json()

    critique = _run_retry(generate, empty_draft, empty_ctx)
    assert critique.severity == Severity.WARNING
    assert attempts == 2


def test_retry_loop_produces_placeholder_after_two_exceptions(empty_draft, empty_ctx):
    def generate(d, c, followup):
        raise RuntimeError("persistent outage")

    critique = _run_retry(generate, empty_draft, empty_ctx)
    assert critique.confidence == 0.0
    assert "persistent outage" in (critique.error or "")


# ─────────────────────────────────────────────────────────────────────
# Registry — four critic roles auto-registered
# ─────────────────────────────────────────────────────────────────────


def test_registry_auto_registers_all_four_critics(pipeline_config):
    registry = RoleRegistry(pipeline_config)
    for role in ("engineering", "security", "operations", "cost"):
        assert registry.has(role) is True


def test_registry_enabled_roles_contains_all_six_seats(pipeline_config):
    registry = RoleRegistry(pipeline_config)
    enabled = set(registry.enabled_roles())
    assert enabled == {
        "product", "engineering", "security",
        "operations", "cost", "judge",
    }


@pytest.mark.parametrize(
    ("cls", "expected_model", "expected_dimension"),
    [
        (EngineeringRole, "claude-sonnet-4-6", CritiqueDimension.FEASIBILITY),
        (SecurityRole, "claude-opus-4-6", CritiqueDimension.RISK_COVERAGE),
        (OperationsRole, None, CritiqueDimension.COMPLETENESS),
        (CostRole, None, CritiqueDimension.FEASIBILITY),
    ],
    ids=["engineering", "security", "operations", "cost"],
)
def test_concrete_critic_role_metadata(cls, expected_model, expected_dimension):
    """Plan's role matrix pins defaults per critic; changing them is
    a deliberate design choice that should fail this test until updated."""

    if expected_model is not None:
        assert cls.default_model == expected_model
    assert cls.default_dimension == expected_dimension


def test_critic_role_instance_calls_injectable_llm(empty_draft, empty_ctx):
    """The base class's default _call_llm raises. Tests inject a stub."""

    role = EngineeringRole()
    role._call_llm = lambda prompt: substantive_critique_json()  # type: ignore[method-assign]
    critique = role.critique(empty_draft, empty_ctx)
    assert critique.author_role == "engineering"
    assert critique.severity == Severity.WARNING


# ─────────────────────────────────────────────────────────────────────
# Parallel fan-out via Send — reuses conftest.stub_critic_registry
# ─────────────────────────────────────────────────────────────────────


def test_fan_out_runs_all_four_critics_per_round(
    fake_refined, pipeline_config, stub_critic_registry,
):
    with patched_product_propose(fake_refined), forced_judge_score(passing=True):
        terminal = run_debate(
            **debate_kwargs(refinery_run_id="refinery-test-fan-out"),
            config=pipeline_config,
            registry=stub_critic_registry,
        )
    trace = terminal["final_trace"]
    round_critiques = trace["critiques_by_round"].get(1, [])
    assert len(round_critiques) == 4
    roles = {c["author_role"] for c in round_critiques}
    assert roles == set(CRITIC_ROLES)


def test_fan_out_tolerates_one_critic_crash(
    fake_refined, pipeline_config, stub_critic_registry,
):
    class _CrashingSec(SecurityRole):
        def _call_llm(self, prompt: str) -> str:
            raise RuntimeError("security llm outage")

    stub_critic_registry.register("security", _CrashingSec)

    with patched_product_propose(fake_refined), forced_judge_score(passing=True):
        terminal = run_debate(
            **debate_kwargs(refinery_run_id="refinery-test-fan-out-crash"),
            config=pipeline_config,
            registry=stub_critic_registry,
        )

    round_critiques = terminal["final_trace"]["critiques_by_round"].get(1, [])
    assert len(round_critiques) == 4
    by_role = {c["author_role"]: c for c in round_critiques}
    assert by_role["security"]["severity"] == "info"
    assert by_role["security"]["error"] is not None
    assert by_role["engineering"]["severity"] == "warning"


def test_anti_collusion_all_rubber_stamp_becomes_all_placeholders(
    fake_refined, pipeline_config, stub_critic_registry,
):
    """Panel-level invariant: if every critic rubber-stamps (twice),
    every slot becomes a placeholder. The round still proceeds."""

    def _rubber_stamp_stub(cls, role_name: str):
        class _RubberStamp(cls):
            def _call_llm(self, prompt: str) -> str:
                return json.dumps({
                    "author_role": role_name, "severity": "warning",
                    "finding": "Looks good to me", "proposed_fix": "",
                })
        return _RubberStamp

    for role, cls in [
        ("engineering", EngineeringRole), ("security", SecurityRole),
        ("operations", OperationsRole), ("cost", CostRole),
    ]:
        stub_critic_registry.register(role, _rubber_stamp_stub(cls, role))

    with patched_product_propose(fake_refined), forced_judge_score(passing=True):
        terminal = run_debate(
            **debate_kwargs(refinery_run_id="refinery-test-anti-collusion"),
            config=pipeline_config,
            registry=stub_critic_registry,
        )

    round_critiques = terminal["final_trace"]["critiques_by_round"].get(1, [])
    assert len(round_critiques) == 4
    for c in round_critiques:
        assert c["severity"] == "info"
        assert c["confidence"] == 0.0
        assert c["error"] is not None


def test_fan_out_on_generator_abort_skips_critics_and_goes_to_finalize(
    pipeline_config, stub_critic_registry, monkeypatch,
):
    def _boom(self, requirement, context):  # noqa: ARG001
        raise RuntimeError("generator outage")

    monkeypatch.setattr(ProductRole, "propose", _boom)
    # Judge.score is never reached on abort, but patch it anyway so
    # the fixture shape matches the other integration tests.
    with forced_judge_score(passing=False):
        terminal = run_debate(
            **debate_kwargs(refinery_run_id="refinery-test-abort-skips-panel"),
            config=pipeline_config,
            registry=stub_critic_registry,
        )

    trace = terminal["final_trace"]
    assert trace["convergence_status"] == ConvergenceStatus.ABORTED.value
    assert trace["critiques_by_round"] in ({}, {1: []})
