"""Shared fixtures for refinery tests.

Previously every test file defined its own near-identical ``_fake_refined``,
``_stub_critic_cls``, and ``_registry_with_stubbed_critics`` helpers. The
duplication ran ~500 lines across the suite and rebuilt the same
``PipelineConfig`` + ``RoleRegistry`` per test. This module consolidates
the shared setup into a handful of fixtures.

Scoping rationale:
- ``pipeline_config`` — module-scoped. ``PipelineConfig()`` is a pure
  Pydantic construction but validates ~40 fields; running it once per
  test file rather than per test saves measurable time.
- ``stub_critic_registry`` — function-scoped because tests mutate
  individual role bindings (e.g. injecting a crashing Security stub).
- ``patched_product_propose`` — function-scoped context manager.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from dark_factory.api.refinery.models import (
    RefinedRequirement,
    RequirementRelationship,
    SuggestedSpec,
)
from dark_factory.api.refinery.roles._shared.critic_role import _CriticBaseRole
from dark_factory.api.refinery.roles.cost import CostRole
from dark_factory.api.refinery.roles.engineering import EngineeringRole
from dark_factory.api.refinery.roles.judge import JudgeRole
from dark_factory.api.refinery.roles.operations import OperationsRole
from dark_factory.api.refinery.roles.product import ProductRole
from dark_factory.api.refinery.roles.registry import RoleRegistry
from dark_factory.api.refinery.roles.security import SecurityRole
from dark_factory.config import PipelineConfig


# ─────────────────────────────────────────────────────────────────────
# Autouse: disable real LLM calls in tests
# ─────────────────────────────────────────────────────────────────────
#
# The production ``_call_llm`` hooks now route through
# ``roles/_shared/llm.py::call_refinery_llm``, which makes a real
# OpenAI Responses call. Tests must never reach that path — without
# this fixture they would hang on network I/O (or burn API credits
# for the suite). Each test that needs LLM-flavored behaviour
# overrides ``_call_llm`` on a specific role; the default here is to
# raise ``NotImplementedError`` so the deterministic fallback inside
# every verb (defend, reconcile, critique, propose) fires.


def _no_llm_stub(self, prompt: str) -> str:  # noqa: ARG001 — signature must match
    raise NotImplementedError(
        "Real LLM call disabled in tests. Override _call_llm on the "
        "concrete role (or use forced_judge_score for Judge.score)."
    )


def _fallback_judge_score(self, draft, context, trace=None):  # noqa: ARG001
    """Default JudgeRole.score for tests — uses FallbackJudge so the
    debate graph runs without DeepEval / OpenAI. Tests that need a
    specific outcome use ``forced_judge_score`` to override on top."""

    from dark_factory.api.refinery.judge.fallback_judge import FallbackJudge

    if not hasattr(self, "_test_fallback_judge"):
        self._test_fallback_judge = FallbackJudge()
    return self._test_fallback_judge.score(draft, context, trace)


@pytest.fixture(autouse=True)
def _disable_real_llm_calls():
    """Block production network paths during tests:

    1. ``_call_llm`` raises ``NotImplementedError`` on every role so
       deterministic fallbacks fire instead of the OpenAI helper.
    2. ``JudgeRole.score`` is replaced with a FallbackJudge call so
       the debate graph's score node doesn't reach DeepEval (which
       blocks on OpenAI auth probing).

    Tests that need a specific outcome still override either hook
    inside the test body (``forced_judge_score`` is the standard one
    for the score path).
    """

    with patch.object(_CriticBaseRole, "_call_llm", _no_llm_stub), \
         patch.object(JudgeRole, "_call_llm", _no_llm_stub), \
         patch.object(ProductRole, "_call_llm", _no_llm_stub), \
         patch.object(JudgeRole, "score", _fallback_judge_score):
        yield


# ─────────────────────────────────────────────────────────────────────
# Config + draft fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline_config() -> PipelineConfig:
    """Module-scoped default config — re-used across tests in one file."""

    return PipelineConfig()


def build_refined_requirement(
    req_id: str = "req-1",
    *,
    title: str = "Refined auth feature",
    description: str | None = None,
    priority: str = "high",
    tags: tuple[str, ...] = ("auth", "security"),
    with_relationships: bool = True,
    with_specs: bool = True,
    changes: tuple[str, ...] = (),
    pass_context: str = "",
) -> RefinedRequirement:
    """Factory for the canonical test RefinedRequirement.

    Default output matches the ``_fake_refined`` that every test file
    was building by hand. Overrides are keyword-only so opinionated
    callers (e.g. the ``_fake_refined_unchanged`` shape from
    ``test_router.py``) stay readable.
    """

    if description is None:
        description = (
            "Given a valid client when authorize is called then a redirect "
            "occurs; supports OAuth2 with 30s timeout and 100 req/s."
        )
    return RefinedRequirement(
        id=req_id,
        original_title="Original",
        original_description="Original description",
        title=title,
        description=description,
        priority=priority,
        tags=list(tags),
        relationships=[
            RequirementRelationship(
                target_id="req-2", type="depends_on",
                rationale="depends on session mgr",
            )
        ] if with_relationships else [],
        suggested_specs=[
            SuggestedSpec(
                title="OAuth2 flow", capability="auth",
                description="Implement OAuth2 authorization-code flow",
                acceptance_criteria=[
                    "GIVEN a valid client WHEN authorize is called THEN a redirect occurs",
                    "GIVEN an invalid code WHEN token is called THEN 400 is returned",
                    "Token endpoint latency < 300ms at 100 req/s",
                ],
            )
        ] if with_specs else [],
        changes=list(changes),
        pass_context=pass_context,
    )


@pytest.fixture
def fake_refined() -> RefinedRequirement:
    """Canonical per-test RefinedRequirement — matches what the old
    per-file ``_fake_refined`` helpers produced."""

    return build_refined_requirement()


# ─────────────────────────────────────────────────────────────────────
# Critic stubs + registry
# ─────────────────────────────────────────────────────────────────────


def substantive_critique_json(
    role: str = "engineering",
    dimension: str = "feasibility",
    severity: str = "warning",
    finding: str | None = None,
    proposed_fix: str | None = None,
) -> str:
    """Return a canned Critique JSON payload that clears the
    rubber-stamp validator."""

    if finding is None:
        finding = (
            "The requirement specifies OAuth2 but doesn't declare the "
            "token store backend, which affects horizontal scalability."
        )
    if proposed_fix is None:
        proposed_fix = "pick Redis or Qdrant-backed token store; document TTLs"
    return json.dumps({
        "author_role": role,
        "dimension": dimension,
        "severity": severity,
        "finding": finding,
        "proposed_fix": proposed_fix,
        "cited_evidence": [],
        "confidence": 0.7,
    })


def stub_critic_cls(role_cls, *, severity: str = "warning"):
    """Produce a critic subclass that returns a substantive canned
    critique via its ``_call_llm`` hook. Replaces the 3 near-identical
    ``_stub_critic_cls`` / ``_stub_cls`` / ``_install_stub`` helpers that
    used to live in each test module."""

    class _Stubbed(role_cls):
        def _call_llm(self, prompt: str) -> str:  # noqa: ARG002 — unused prompt
            return json.dumps({
                "author_role": self.role_name,
                "severity": severity,
                "dimension": self.default_dimension.value,
                "finding": (
                    f"{self.role_name} finds an architectural concern "
                    "worth addressing in a future iteration."
                ),
                "proposed_fix": f"apply the {self.role_name} checklist",
            })

    return _Stubbed


_CRITIC_CLASSES = (
    ("engineering", EngineeringRole),
    ("security", SecurityRole),
    ("operations", OperationsRole),
    ("cost", CostRole),
)


def make_stub_critic_registry(
    cfg: PipelineConfig | None = None,
    *,
    severity: str = "warning",
) -> RoleRegistry:
    """Build a RoleRegistry with all 4 critic roles stubbed to return
    substantive critiques. Tests that need a crashing or
    rubber-stamping variant replace specific entries via
    ``registry.register(role, CustomStub)`` after construction."""

    registry = RoleRegistry(cfg or PipelineConfig())
    for role_name, role_cls in _CRITIC_CLASSES:
        registry.register(role_name, stub_critic_cls(role_cls, severity=severity))
    return registry


@pytest.fixture
def stub_critic_registry(pipeline_config: PipelineConfig) -> RoleRegistry:
    """Function-scoped registry with all 4 critics stubbed. Mutate
    individual roles in tests via ``stub_critic_registry.register(...)``."""

    return make_stub_critic_registry(pipeline_config)


# ─────────────────────────────────────────────────────────────────────
# Product LLM stub + run_debate kwargs helper
# ─────────────────────────────────────────────────────────────────────


@contextmanager
def patched_product_propose(fake: RefinedRequirement, memories: list | None = None):
    """Patch ``ProductRole.propose`` to return a Draft built from
    ``fake`` and propagate ``memories`` through the
    ``suggested_memories_accumulator`` evidence-bag entry.

    Tests that need to assert exact LLM-call arguments should patch
    ``ProductRole._call_llm`` directly with a canned JSON return.
    """

    from dark_factory.api.refinery.roles.product import (
        ProductRole,
        _refined_to_draft,
    )

    mems = list(memories or [])

    def _side_effect(self, requirement, context):  # noqa: ARG001 — bound method
        ev = context.evidence or {}
        buf = ev.get("suggested_memories_accumulator")
        if isinstance(buf, list) and mems:
            buf.extend(mems)
        return _refined_to_draft(fake, produced_by=self.role_name)

    # ``autospec=True`` makes the mock respect the bound-method
    # signature, so calls land as ``mock(self, req, ctx)`` and tests
    # can use ``mock.assert_called_once()`` / inspect ``call_args``.
    with patch.object(ProductRole, "propose", autospec=True) as mock:
        mock.side_effect = _side_effect
        yield mock


@pytest.fixture
def patched_runner(fake_refined: RefinedRequirement):
    """Fixture form of ``patched_product_propose`` for tests that don't
    need to customise the fake."""

    with patched_product_propose(fake_refined) as mock:
        yield mock


@contextmanager
def forced_judge_score(passing: bool = False):
    """Patch ``JudgeRole.score`` to a deterministic stub.

    Integration tests that don't care about the scoring layer (e.g.
    fan-out smoke tests) use this to short-circuit ``CombinedJudgePipeline``
    and avoid hitting DeepEval's OpenAI auth probe.
    """

    from dark_factory.api.refinery.contracts import (
        CritiqueDimension,
        EvaluationScore,
    )
    from dark_factory.api.refinery.roles.judge import JudgeRole

    target = 0.95 if passing else 0.4

    def _stub(self, draft, context, trace=None):  # noqa: ARG001
        return EvaluationScore(
            dimensions={d.value: target for d in CritiqueDimension},
            dimensions_semantic_raw={d.value: target for d in CritiqueDimension},
            reasons={}, overall=target, passed=passing,
            aggregation="min",
            model_used="stub-pass" if passing else "stub-fail",
        )

    with patch.object(JudgeRole, "score", _stub):
        yield


def debate_kwargs(
    *,
    req_id: str = "req-1",
    title: str = "T",
    description: str = "D",
    priority: str = "medium",
    tags: tuple[str, ...] = (),
    max_rounds: int = 3,
    score_threshold: float = 0.8,
    research_call_cap: int = 1,
    escalation_cap: int = 1,
    base_model: str = "gpt-5.4",
    strong_model: str = "gpt-5.4",
    reasoning_effort: str = "xhigh",
    timeout_seconds: float = 60.0,
    tmpdir: str = "/tmp/refinery-test",
    refinery_run_id: str = "refinery-test",
    **extra,
) -> dict:
    """Build a kwargs dict for ``run_debate`` with sensible defaults.

    Collapses ~18 lines of inline kwargs-per-test into one call:
    ``run_debate(**debate_kwargs(max_rounds=2, escalation_cap=0))``.
    """

    return {
        "req": {"id": req_id, "title": title, "description": description,
                "priority": priority, "tags": list(tags)},
        "all_requirements": [],
        "run_context": None,
        "tmpdir": tmpdir,
        "max_rounds": max_rounds,
        "score_threshold": score_threshold,
        "research_call_cap": research_call_cap,
        "escalation_cap": escalation_cap,
        "base_model": base_model,
        "strong_model": strong_model,
        "reasoning_effort": reasoning_effort,
        "timeout_seconds": timeout_seconds,
        "on_progress": None,
        "refinery_run_id": refinery_run_id,
        **extra,
    }


# ─────────────────────────────────────────────────────────────────────
# SSE / async-generator test helpers
# ─────────────────────────────────────────────────────────────────────


def make_stub_request(metrics_client: object | None = None):
    """Build a minimal request object that ``run_refinery_stream``
    accepts. ``metrics_client`` controls whether the resume registry
    is enabled in the test — pass ``None`` for the resume-disabled
    path, anything truthy to exercise registry construction.

    Replaces the duplicated _StubAppState/_StubApp/_StubRequest
    classes that lived in test_resume.py + test_cross_set_review.py.
    """

    class _StubAppState:
        pass

    state = _StubAppState()
    state.metrics_client = metrics_client  # type: ignore[attr-defined]
    state.neo4j_client = None  # type: ignore[attr-defined]
    state.memory_repo = None  # type: ignore[attr-defined]

    class _StubApp:
        pass

    app = _StubApp()
    app.state = state  # type: ignore[attr-defined]

    class _StubRequest:
        pass

    request = _StubRequest()
    request.app = app  # type: ignore[attr-defined]
    return request


def drive_async_gen(gen) -> list[dict]:
    """Drain an async generator to a list. Used by SSE tests that
    don't care about back-pressure timing."""

    import asyncio

    async def _run():
        out: list[dict] = []
        async for ev in gen:
            out.append(ev)
        return out

    return asyncio.run(_run())
