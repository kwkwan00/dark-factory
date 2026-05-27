"""Tests for the spec reconciliation stage."""

from __future__ import annotations

from unittest.mock import MagicMock

from dark_factory.models.domain import PipelineContext, Requirement, Spec
from dark_factory.stages.spec_reconciliation import (
    SpecReconciliationStage,
    _ReconciliationResult,
    _SpecPatch,
    _ReconciliationIssue,
    _detect_cycles,
    _find_uncovered_requirements,
    _remove_back_edges,
    _strip_phantom_dependency_ids,
    _strip_phantom_requirement_ids,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _req(id: str, title: str = "") -> Requirement:
    return Requirement(
        id=id, title=title or id, description=f"Desc for {id}",
        source_file="test.md",
    )


def _spec(
    id: str,
    req_ids: list[str] | None = None,
    deps: list[str] | None = None,
    title: str = "",
) -> Spec:
    return Spec(
        id=id,
        title=title or f"Spec {id}",
        description=f"Description of {id}",
        requirement_ids=req_ids or [],
        dependencies=deps or [],
    )


# ── Deterministic checks ─────────────────────────────────────────────


class TestStripPhantomRequirementIds:
    def test_removes_invalid_ids(self):
        specs = [_spec("s1", req_ids=["r1", "r-ghost", "r2"])]
        removed = _strip_phantom_requirement_ids(specs, {"r1", "r2"})
        assert removed == 1
        assert specs[0].requirement_ids == ["r1", "r2"]

    def test_no_op_when_all_valid(self):
        specs = [_spec("s1", req_ids=["r1"])]
        removed = _strip_phantom_requirement_ids(specs, {"r1"})
        assert removed == 0
        assert specs[0].requirement_ids == ["r1"]


class TestStripPhantomDependencyIds:
    def test_removes_nonexistent_deps(self):
        specs = [
            _spec("s1", deps=["s2", "s-ghost"]),
            _spec("s2"),
        ]
        removed = _strip_phantom_dependency_ids(specs)
        assert removed == 1
        assert specs[0].dependencies == ["s2"]

    def test_no_op_when_all_valid(self):
        specs = [_spec("s1", deps=["s2"]), _spec("s2")]
        removed = _strip_phantom_dependency_ids(specs)
        assert removed == 0


class TestDetectCycles:
    def test_no_cycle(self):
        specs = [
            _spec("s1", deps=["s2"]),
            _spec("s2", deps=["s3"]),
            _spec("s3"),
        ]
        assert _detect_cycles(specs) == []

    def test_simple_cycle(self):
        specs = [
            _spec("s1", deps=["s2"]),
            _spec("s2", deps=["s1"]),
        ]
        back_edges = _detect_cycles(specs)
        assert len(back_edges) == 1
        # One of the two edges is the back-edge
        assert set(back_edges[0]) == {"s1", "s2"}

    def test_three_node_cycle(self):
        specs = [
            _spec("s1", deps=["s2"]),
            _spec("s2", deps=["s3"]),
            _spec("s3", deps=["s1"]),
        ]
        back_edges = _detect_cycles(specs)
        assert len(back_edges) == 1

    def test_self_loop(self):
        specs = [_spec("s1", deps=["s1"])]
        back_edges = _detect_cycles(specs)
        assert len(back_edges) == 1
        assert back_edges[0] == ("s1", "s1")


class TestRemoveBackEdges:
    def test_removes_cycle_edge(self):
        specs = [
            _spec("s1", deps=["s2"]),
            _spec("s2", deps=["s1"]),
        ]
        back_edges = _detect_cycles(specs)
        removed = _remove_back_edges(specs, back_edges)
        assert removed == 1
        # After removal, no cycles should remain
        assert _detect_cycles(specs) == []


class TestFindUncoveredRequirements:
    def test_all_covered(self):
        reqs = [_req("r1"), _req("r2")]
        specs = [_spec("s1", req_ids=["r1"]), _spec("s2", req_ids=["r2"])]
        assert _find_uncovered_requirements(reqs, specs) == []

    def test_uncovered_detected(self):
        reqs = [_req("r1"), _req("r2"), _req("r3")]
        specs = [_spec("s1", req_ids=["r1"])]
        uncovered = _find_uncovered_requirements(reqs, specs)
        assert set(uncovered) == {"r2", "r3"}

    def test_empty_specs(self):
        reqs = [_req("r1")]
        uncovered = _find_uncovered_requirements(reqs, [])
        assert uncovered == ["r1"]


# ── Full stage tests ─────────────────────────────────────────────────


class TestSpecReconciliationStage:
    def test_no_specs_returns_early(self):
        ctx = PipelineContext(requirements=[_req("r1")])
        result = SpecReconciliationStage(llm=None).run(ctx)
        assert result.specs == []

    def test_deterministic_fixes_without_llm(self):
        """Stage fixes phantom refs and cycles even without an LLM."""
        reqs = [_req("r1"), _req("r2")]
        specs = [
            _spec("s1", req_ids=["r1", "r-phantom"], deps=["s2"]),
            _spec("s2", req_ids=["r2"], deps=["s1"]),  # cycle with s1
        ]
        ctx = PipelineContext(requirements=reqs, specs=specs)
        result = SpecReconciliationStage(llm=None).run(ctx)

        # Phantom requirement removed
        assert "r-phantom" not in result.specs[0].requirement_ids
        assert "r1" in result.specs[0].requirement_ids

        # Cycle broken (one of the edges removed)
        all_deps = set()
        for s in result.specs:
            for d in s.dependencies:
                all_deps.add((s.id, d))
        # Should no longer have both directions
        assert not (("s1", "s2") in all_deps and ("s2", "s1") in all_deps)

    def test_llm_patches_applied(self):
        """LLM-suggested dependency additions are applied."""
        reqs = [_req("r1"), _req("r2")]
        specs = [
            _spec("s1", req_ids=["r1"]),
            _spec("s2", req_ids=["r2"]),
        ]

        llm_result = _ReconciliationResult(
            specs=[
                _SpecPatch(spec_id="s1", requirement_ids=["r1"], dependencies=["s2"]),
                _SpecPatch(spec_id="s2", requirement_ids=["r2"], dependencies=[]),
            ],
            issues=[
                _ReconciliationIssue(
                    severity="warning", spec_id="s1",
                    message="s1 implicitly depends on s2",
                ),
            ],
        )

        mock_llm = MagicMock()
        mock_llm.complete_structured.return_value = llm_result

        ctx = PipelineContext(requirements=reqs, specs=specs)
        result = SpecReconciliationStage(llm=mock_llm).run(ctx)

        # LLM-suggested dep should be added
        assert "s2" in result.specs[0].dependencies

    def test_llm_failure_falls_back_to_deterministic(self):
        """Stage still runs deterministic fixes when LLM fails."""
        reqs = [_req("r1")]
        specs = [_spec("s1", req_ids=["r1", "r-phantom"])]

        mock_llm = MagicMock()
        mock_llm.complete_structured.side_effect = RuntimeError("LLM down")

        ctx = PipelineContext(requirements=reqs, specs=specs)
        result = SpecReconciliationStage(llm=mock_llm).run(ctx)

        # Phantom still removed despite LLM failure
        assert result.specs[0].requirement_ids == ["r1"]

    def test_llm_cannot_introduce_cycles(self):
        """If LLM adds deps that create a cycle, the cycle is broken."""
        reqs = [_req("r1"), _req("r2")]
        specs = [
            _spec("s1", req_ids=["r1"], deps=["s2"]),
            _spec("s2", req_ids=["r2"]),
        ]

        # LLM adds s1 as dep of s2, creating a cycle
        llm_result = _ReconciliationResult(
            specs=[
                _SpecPatch(spec_id="s1", requirement_ids=["r1"], dependencies=["s2"]),
                _SpecPatch(spec_id="s2", requirement_ids=["r2"], dependencies=["s1"]),
            ],
            issues=[],
        )

        mock_llm = MagicMock()
        mock_llm.complete_structured.return_value = llm_result

        ctx = PipelineContext(requirements=reqs, specs=specs)
        result = SpecReconciliationStage(llm=mock_llm).run(ctx)

        # Cycle must be broken
        assert _detect_cycles(result.specs) == []

    def test_llm_cannot_add_phantom_deps(self):
        """LLM-suggested deps pointing to nonexistent specs are stripped."""
        reqs = [_req("r1")]
        specs = [_spec("s1", req_ids=["r1"])]

        llm_result = _ReconciliationResult(
            specs=[
                _SpecPatch(
                    spec_id="s1",
                    requirement_ids=["r1"],
                    dependencies=["s-hallucinated"],
                ),
            ],
            issues=[],
        )

        mock_llm = MagicMock()
        mock_llm.complete_structured.return_value = llm_result

        ctx = PipelineContext(requirements=reqs, specs=specs)
        result = SpecReconciliationStage(llm=mock_llm).run(ctx)

        # Hallucinated dep must not survive
        assert result.specs[0].dependencies == []

    def test_llm_adds_missing_requirement_coverage(self):
        """LLM can add a requirement_id to a spec that was missing it."""
        reqs = [_req("r1"), _req("r2")]
        specs = [
            _spec("s1", req_ids=["r1"]),
            _spec("s2", req_ids=[]),  # missing r2
        ]

        llm_result = _ReconciliationResult(
            specs=[
                _SpecPatch(spec_id="s1", requirement_ids=["r1"], dependencies=[]),
                _SpecPatch(spec_id="s2", requirement_ids=["r2"], dependencies=[]),
            ],
            issues=[],
        )

        mock_llm = MagicMock()
        mock_llm.complete_structured.return_value = llm_result

        ctx = PipelineContext(requirements=reqs, specs=specs)
        result = SpecReconciliationStage(llm=mock_llm).run(ctx)

        # r2 should now be covered
        assert "r2" in result.specs[1].requirement_ids
        assert _find_uncovered_requirements(reqs, result.specs) == []


class TestSettingsToggle:
    def test_setting_exists(self):
        from dark_factory.config import PipelineConfig

        ps = PipelineConfig()
        assert ps.enable_spec_reconciliation is True

    def test_setting_can_be_disabled(self):
        from dark_factory.config import PipelineConfig

        ps = PipelineConfig(enable_spec_reconciliation=False)
        assert ps.enable_spec_reconciliation is False
