"""Foundation smoke tests — contracts, ABCs, config defaults.

Proves the refinery module hierarchy imports cleanly and the
PipelineConfig defaults still match expectations. Concrete role,
graph, judge, and research behaviour tests live in their own modules.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dark_factory.api.refinery import roles as refinery_roles
from dark_factory.api.refinery.context.base import ContextBuilder, RoleFilterPolicy, RunContext
from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    Critique,
    CritiqueDimension,
    Draft,
    EvaluationScore,
    MemoryKind,
    RawRequirement,
    Rebuttal,
    RebuttalEntry,
    RoleContext,
    Severity,
    SourceTier,
    SuggestedMemoryV2,
    ValidationStatus,
)
from dark_factory.api.refinery.judge.base import Judge
from dark_factory.api.refinery.observability import (
    CallRecord,
    DebateTrace,
    RoundRecord,
    TraceContext,
)
from dark_factory.api.refinery.research.base import ResearchProvider, TierBudgetExhausted
from dark_factory.api.refinery.roles import RoleAgent, RoleRegistry
from dark_factory.config import PipelineConfig


# ─────────────────────────────────────────────────────────────────────
# Config — defaults
# ─────────────────────────────────────────────────────────────────────


def test_refinery_config_defaults_are_reasonable(pipeline_config):
    cfg = pipeline_config
    assert cfg.refinery_debate_max_rounds == 3
    assert 0.0 <= cfg.refinery_debate_threshold <= 1.0
    assert cfg.refinery_research_cap >= 0
    assert cfg.refinery_escalation_cap >= 0
    assert "product" in cfg.refinery_roles_enabled
    assert "judge" in cfg.refinery_roles_enabled
    # 6-seat adversarial panel + Research-as-capability — Research is not a seat.
    assert "research" not in cfg.refinery_roles_enabled
    assert cfg.refinery_rules_enabled is True
    assert set(cfg.refinery_judge_thresholds) == {
        "clarity", "testability", "feasibility", "completeness", "risk_coverage",
        "reversibility",
    }


def test_swarm_memory_recall_scoped_to_legacy_kinds(pipeline_config):
    """Swarm must stay insulated from refinery-kind memories."""

    assert set(pipeline_config.swarm_memory_kinds_enabled) == {
        "pattern", "mistake", "solution", "strategy",
    }


# ─────────────────────────────────────────────────────────────────────
# Contracts — Pydantic shapes round-trip
# ─────────────────────────────────────────────────────────────────────


def test_raw_requirement_roundtrip():
    req = RawRequirement(id="req-1", title="T", description="D")
    assert req.model_dump()["id"] == "req-1"


def test_draft_carries_convergence_fields():
    draft = Draft(
        requirement_id="req-1",
        title="T",
        description="D",
        priority="medium",
        produced_by="product",
        convergence_status=ConvergenceStatus.CONVERGED,
        explicit_tradeoffs=["we preferred latency over cost"],
    )
    assert draft.convergence_status == ConvergenceStatus.CONVERGED
    assert draft.explicit_tradeoffs == ["we preferred latency over cost"]


def test_critique_enum_fields_type_checked():
    c = Critique(
        author_role="security",
        dimension=CritiqueDimension.RISK_COVERAGE,
        severity=Severity.BLOCKER,
        finding="missing threat model for OAuth flow",
        proposed_fix="require explicit threat model with STRIDE analysis",
    )
    assert c.dimension == CritiqueDimension.RISK_COVERAGE
    assert c.severity == Severity.BLOCKER


def test_role_context_is_frozen():
    ctx = RoleContext(role="product", requirement_id="req-1")
    with pytest.raises(ValidationError):
        ctx.role = "security"  # type: ignore[misc]


def test_rebuttal_wraps_revised_draft():
    draft = Draft(requirement_id="req-1", title="T", description="D",
                  priority="low", produced_by="judge", iteration=1)
    reb = Rebuttal(
        revised_draft=draft,
        entries=[RebuttalEntry(critique_ref="c-0", action="accepted",
                               rationale="applied the proposed fix verbatim")],
    )
    assert reb.revised_draft.iteration == 1
    assert reb.mode == "synthesize"  # default


def test_evaluation_score_tracks_both_pre_and_post_penalty_dims():
    score = EvaluationScore(
        dimensions={"clarity": 0.5},              # post-penalty
        dimensions_semantic_raw={"clarity": 0.9},  # pre-penalty LLM
    )
    assert score.dimensions["clarity"] == 0.5
    assert score.dimensions_semantic_raw["clarity"] == 0.9


def test_suggested_memory_v2_supports_all_curated_kinds():
    for kind in MemoryKind:
        m = SuggestedMemoryV2(
            kind=kind,
            summary=f"summary for {kind.value}",
            body=f"body for {kind.value}",
            source_role="judge",
            validation_status=ValidationStatus.VALIDATED,
        )
        assert m.kind == kind


def test_source_tier_ordering_matches_trust():
    # T0 is the most trusted; T5 is the least. Plan's trust-weight table.
    assert SourceTier.T0_STRUCTURED.value < SourceTier.T5_WEB.value


# ─────────────────────────────────────────────────────────────────────
# ABCs — can be imported; defaults raise NotImplementedError
# ─────────────────────────────────────────────────────────────────────


def test_role_agent_verbs_default_to_not_implemented():
    class EmptyRole(RoleAgent):
        role_name = "empty"

    r = EmptyRole()
    with pytest.raises(NotImplementedError):
        r.propose(RawRequirement(id="x", title="t", description="d"),
                  RoleContext(role="empty", requirement_id="x"))
    with pytest.raises(NotImplementedError):
        r.critique(
            Draft(requirement_id="x", title="t", description="d",
                  priority="low", produced_by="empty"),
            RoleContext(role="empty", requirement_id="x"),
        )


def test_judge_is_abstract():
    with pytest.raises(TypeError):
        Judge()  # type: ignore[abstract]


def test_context_builder_is_abstract():
    with pytest.raises(TypeError):
        ContextBuilder()  # type: ignore[abstract]


def test_research_provider_is_abstract():
    with pytest.raises(TypeError):
        ResearchProvider()  # type: ignore[abstract]


def test_tier_budget_exhausted_is_exception():
    assert issubclass(TierBudgetExhausted, Exception)


# ─────────────────────────────────────────────────────────────────────
# RoleRegistry — empty-by-default, register/get workflow
# ─────────────────────────────────────────────────────────────────────


def test_role_registry_auto_registers_shipped_roles_and_respects_register(pipeline_config):
    registry = RoleRegistry(pipeline_config)

    # The six adversarial panel seats are auto-registered; Research is
    # a capability invoked by the debate router, never a panel seat.
    for role in ("product", "judge", "engineering", "security",
                 "operations", "cost"):
        assert registry.has(role) is True, f"{role} should be auto-registered"
    assert registry.has("research") is False
    with pytest.raises(KeyError):
        registry.get("research")

    # A test can still swap an existing factory for a stub.
    class StubRole(RoleAgent):
        role_name = "security"
        default_model = "stub-model"

    registry.register("security", StubRole)
    instance = registry.get("security")
    assert isinstance(instance, RoleAgent)
    assert instance.role_name == "security"


def test_role_registry_applies_model_override():
    cfg = PipelineConfig(refinery_role_models={"security": "claude-opus-4-7"})
    registry = RoleRegistry(cfg)

    class StubSec(RoleAgent):
        role_name = "security"
        default_model = "claude-sonnet-4-6"

    registry.register("security", StubSec)
    instance = registry.get("security")
    # Override lands on the private channel — class metadata stays as
    # documentation of the default.
    assert instance._model == "claude-opus-4-7"  # type: ignore[attr-defined]
    assert StubSec.default_model == "claude-sonnet-4-6"


def test_role_registry_enabled_roles_filters_to_registered(pipeline_config):
    registry = RoleRegistry(pipeline_config)
    # All six seats of the adversarial panel are registered after Phase 5.
    # Research is a capability (invoked by the debate router in Phase 8),
    # not a panel seat, so it's never in enabled_roles().
    assert set(registry.enabled_roles()) == {
        "product", "engineering", "security",
        "operations", "cost", "judge",
    }


# ─────────────────────────────────────────────────────────────────────
# Observability — DebateTrace round-trip + TraceContext record_call
# ─────────────────────────────────────────────────────────────────────


def test_debate_trace_serialises_with_empty_rounds():
    t = DebateTrace(requirement_id="req-1", refinery_run_id="refinery-test")
    assert t.model_dump()["requirement_id"] == "req-1"
    assert t.rounds == []


def test_trace_context_record_call_appends_on_success():
    trace = DebateTrace(requirement_id="req-1", refinery_run_id="refinery-test")
    ctx = TraceContext(trace=trace)
    with ctx.record_call(role="product", kind="propose", model="gpt-5.4") as rec:
        rec.tokens_in = 42
        rec.tokens_out = 17
    assert rec.role == "product"
    assert rec.tokens_in == 42
    assert rec.ended_at is not None


def test_trace_context_record_call_captures_errors():
    trace = DebateTrace(requirement_id="req-1", refinery_run_id="refinery-test")
    ctx = TraceContext(trace=trace)
    with pytest.raises(RuntimeError):
        with ctx.record_call(role="security", kind="critique") as rec:
            raise RuntimeError("llm timed out")
    assert rec.error == "llm timed out"


# ─────────────────────────────────────────────────────────────────────
# RoleFilterPolicy — frozen + defaults
# ─────────────────────────────────────────────────────────────────────


def test_role_filter_policy_is_frozen():
    p = RoleFilterPolicy(memory_kinds=["pattern"], row_cap=5)
    with pytest.raises(ValidationError):
        p.row_cap = 999  # type: ignore[misc]


def test_role_filter_policy_defaults():
    p = RoleFilterPolicy()
    # None values mean "no filter" — the translator emits an empty Qdrant
    # Filter / unconstrained WHERE in that case. The research / judge
    # empty-by-design behaviour is gated on p.empty, NOT on memory_kinds.
    assert p.memory_kinds is None
    assert p.row_cap == 12
    assert p.empty is False


def test_run_context_default_source_mode_empty():
    rc = RunContext()
    assert rc.source_mode == ""
    assert rc.evidence == {}


# ─────────────────────────────────────────────────────────────────────
# Package importability — the roles package re-exports only ABC + registry
# ─────────────────────────────────────────────────────────────────────


def test_roles_package_hides_concrete_classes():
    # Parnas discipline: only RoleAgent and RoleRegistry are public.
    # Concrete roles live in roles/<name>/ and must be imported only by
    # the registry's internal factories.
    public = set(dir(refinery_roles)) - set(
        n for n in dir(refinery_roles) if n.startswith("_")
    )
    # Whatever leaks must be one of the known-safe names. ABC + registry
    # + the package dunder symbol names only.
    assert "RoleAgent" in public
    assert "RoleRegistry" in public
    concretes = {"ProductRole", "EngineeringRole", "SecurityRole",
                 "OperationsRole", "CostRole", "JudgeRole"}
    assert not (public & concretes)
