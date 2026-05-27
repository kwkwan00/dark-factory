"""ProductRole + Phase-2 runner tests.

Proves:
- ProductRole.propose calls the legacy runner and returns a Draft.
- run_phase2_generator produces a RefinedRequirement byte-identical to
  the legacy path on the same inputs.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dark_factory.api.refinery.contracts import (
    Draft,
    DraftRelationship,
    DraftSpec,
    MemoryKind,
    RawRequirement,
    RoleContext,
)
from dark_factory.api.refinery.debate.runner import (
    _draft_to_refined,
    run_phase2_generator,
)
from dark_factory.api.refinery.models import (
    RefinedRequirement,
    SuggestedMemory,
)
from dark_factory.api.refinery.roles.product import (
    ProductRole,
    _refined_to_draft,
    legacy_memory_to_v2,
)
from dark_factory.api.refinery.roles.registry import RoleRegistry
from dark_factory.config import PipelineConfig

from tests.refinery.conftest import forced_judge_score, patched_product_propose


# ─────────────────────────────────────────────────────────────────────
# ProductRole — unit tests
# ─────────────────────────────────────────────────────────────────────


def _empty_draft() -> Draft:
    return Draft(
        requirement_id="r", title="t", description="d",
        priority="low", produced_by="product",
    )


def _empty_ctx() -> RoleContext:
    return RoleContext(role="product", requirement_id="r")


@pytest.mark.parametrize(
    "verb",
    ["critique", "defend", "score"],
    ids=["critique", "defend", "score"],
)
def test_product_role_is_generator_only(verb):
    """propose is the only verb Product overrides; the rest raise."""

    role = ProductRole()
    with pytest.raises(NotImplementedError):
        fn = getattr(role, verb)
        if verb == "defend":
            fn(_empty_draft(), [], _empty_ctx())
        else:
            fn(_empty_draft(), _empty_ctx())


def test_product_role_defaults_match_legacy_refinery():
    """Default model / effort equal today's ``refinery_model`` /
    ``refinery_reasoning_effort``."""

    role = ProductRole()
    assert role.default_model == "gpt-5.4"
    assert role.default_reasoning_effort == "xhigh"
    assert role.model == "gpt-5.4"
    assert role.reasoning_effort == "xhigh"


def test_product_role_propose_returns_draft_built_from_refined(fake_refined):
    role = ProductRole()
    raw = RawRequirement(
        id="req-1", title="Original", description="Original description",
        priority="medium", tags=[],
    )
    with patched_product_propose(fake_refined) as mock_propose:
        ctx = RoleContext(
            role="product", requirement_id="req-1",
            evidence={
                "all_requirements": [], "run_context": None,
                "tmpdir": "/tmp/refinery-test", "max_turns": 5,
                "timeout_seconds": 60.0, "on_progress": None,
            },
        )
        draft = role.propose(raw, ctx)

    mock_propose.assert_called_once()
    assert isinstance(draft, Draft)
    assert draft.requirement_id == "req-1"
    assert draft.title == fake_refined.title
    assert draft.produced_by == "product"
    assert draft.iteration == 0
    assert len(draft.suggested_specs) == 1
    assert draft.suggested_specs[0].capability == "auth"
    assert draft.relationships[0].target_id == "req-2"


def test_product_role_propose_appends_to_memory_accumulator(fake_refined):
    role = ProductRole()
    raw = RawRequirement(id="req-1", title="T", description="D", priority="medium")
    mems = [SuggestedMemory(type="pattern", description="reuse OAuth helper")]
    buf: list[SuggestedMemory] = []

    with patched_product_propose(fake_refined, memories=mems):
        ctx = RoleContext(
            role="product", requirement_id="req-1",
            evidence={"suggested_memories_accumulator": buf},
        )
        role.propose(raw, ctx)

    assert buf == mems


def test_refined_to_draft_roundtrip_preserves_fields(fake_refined):
    draft = _refined_to_draft(fake_refined, produced_by="product")
    assert draft.requirement_id == fake_refined.id
    assert draft.title == fake_refined.title
    assert draft.description == fake_refined.description
    assert draft.priority == fake_refined.priority
    assert list(draft.tags) == list(fake_refined.tags)
    assert [r.target_id for r in draft.relationships] == [
        r.target_id for r in fake_refined.relationships
    ]
    assert [s.title for s in draft.suggested_specs] == [
        s.title for s in fake_refined.suggested_specs
    ]


def test_legacy_memory_upgrade_maps_to_pattern_kind():
    legacy = SuggestedMemory(
        type="strategy",
        description="When refactoring auth, extract a session manager first.",
        context="Auth rewrites",
        applicability="Any auth/session feature",
        rationale="Keeps the rewrite scoped",
    )
    v2 = legacy_memory_to_v2(legacy, source_role="product")
    assert v2.kind == MemoryKind.PATTERN
    assert v2.source_role == "product"
    assert v2.body == legacy.description
    assert v2.context == legacy.context


# ─────────────────────────────────────────────────────────────────────
# run_phase2_generator — full shape of the Phase-2 path
# ─────────────────────────────────────────────────────────────────────


def _runner_kwargs(
    *, max_turns: int = 5, timeout_seconds: float = 60.0,
    model: str = "gpt-5.4", reasoning_effort: str = "xhigh",
) -> dict:
    return {
        "req": {"id": "req-1", "title": "Original",
                "description": "Original description",
                "priority": "medium", "tags": []},
        "all_requirements": [],
        "run_context": None,
        "tmpdir": "/tmp/refinery-test",
        "max_turns": max_turns,
        "timeout_seconds": timeout_seconds,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "on_progress": None,
    }


def test_run_phase2_generator_returns_refined_and_memories(fake_refined, pipeline_config):
    with patched_product_propose(fake_refined), forced_judge_score(passing=True):
        refined, memories, _ = run_phase2_generator(
            **_runner_kwargs(), config=pipeline_config,
        )
    assert isinstance(refined, RefinedRequirement)
    assert refined.id == "req-1"
    assert refined.title == fake_refined.title
    # changes / pass_context stay empty when synthesis is a pass-through.
    assert refined.changes == []
    assert refined.pass_context == ""
    assert memories == []


def test_run_phase2_generator_packs_model_and_effort_into_debate_config(
    fake_refined, pipeline_config,
):
    """``run_phase2_generator`` packs the caller's model + effort into
    a ``DebateConfig`` that the graph forwards to ProductRole. Verify
    by intercepting the configure() call on the role instance."""

    captured: dict[str, object] = {}

    def _intercepted_configure(self, *, model=None, reasoning_effort=None):
        captured["model"] = model
        captured["reasoning_effort"] = reasoning_effort
        if model is not None:
            self._model = model
        if reasoning_effort is not None:
            self._reasoning_effort = reasoning_effort
        return self

    from dark_factory.api.refinery.roles.product import ProductRole

    with patched_product_propose(fake_refined), forced_judge_score(passing=True), \
         patch.object(ProductRole, "configure", _intercepted_configure):
        run_phase2_generator(
            **_runner_kwargs(
                max_turns=7, timeout_seconds=120.0,
                model="gpt-5.4-super", reasoning_effort="low",
            ),
            config=pipeline_config,
        )

    assert captured["model"] == "gpt-5.4-super"
    assert captured["reasoning_effort"] == "low"


def test_run_phase2_generator_raises_when_product_unregistered(
    monkeypatch, pipeline_config,
):
    """The registry MUST contain product + judge — the legacy fallback
    has been removed, so a misconfigured registry should fail loud."""

    monkeypatch.setattr(
        "dark_factory.api.refinery.roles.registry.RoleRegistry._register_defaults",
        lambda self: None,
    )

    with pytest.raises(RuntimeError, match="product"):
        run_phase2_generator(**_runner_kwargs(), config=pipeline_config)


# ─────────────────────────────────────────────────────────────────────
# Registry — overrides land on the instance
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("config_kwargs", "attr", "expected"),
    [
        ({"refinery_role_models": {"product": "claude-opus-4-7"}},
         "_model", "claude-opus-4-7"),
        ({"refinery_role_reasoning_effort": {"product": "medium"}},
         "_reasoning_effort", "medium"),
    ],
    ids=["model_override", "effort_override"],
)
def test_registry_applies_product_overrides(config_kwargs, attr, expected):
    cfg = PipelineConfig(**config_kwargs)
    product = RoleRegistry(cfg).get("product")
    assert getattr(product, attr) == expected


# ─────────────────────────────────────────────────────────────────────
# Inverse mapping — Draft → RefinedRequirement
# ─────────────────────────────────────────────────────────────────────


def test_draft_to_refined_preserves_original_fields():
    raw = RawRequirement(
        id="req-1", title="Original title", description="Original description",
        priority="medium", tags=["legacy"],
    )
    draft = Draft(
        requirement_id="req-1", title="New title", description="New description",
        priority="high", tags=["refined", "auth"],
        suggested_specs=[
            DraftSpec(title="S", capability="c", description="d",
                      acceptance_criteria=["ac1"]),
        ],
        relationships=[
            DraftRelationship(target_id="req-2", type="depends_on", rationale="foo"),
        ],
        produced_by="product",
    )
    refined = _draft_to_refined(draft, raw)
    assert refined.id == "req-1"
    assert refined.original_title == "Original title"
    assert refined.original_description == "Original description"
    assert refined.title == "New title"
    assert refined.description == "New description"
    assert refined.priority == "high"
    assert refined.tags == ["refined", "auth"]
    assert len(refined.suggested_specs) == 1
    assert refined.suggested_specs[0].capability == "c"
    assert refined.relationships[0].target_id == "req-2"
