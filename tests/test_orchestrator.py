"""Tests for the per-feature orchestrator."""

from __future__ import annotations

import pytest

from dark_factory.agents.orchestrator import (
    OrchestratorState,
    aggregate_node,
    check_done,
    topological_layers,
)
from dark_factory.agents.swarm import FeatureResult


# ── topological_layers tests ─────────────────────────────────────────


def test_topological_layers_linear() -> None:
    groups = {"A": ["s1"], "B": ["s2"], "C": ["s3"]}
    deps = {"B": {"A"}, "C": {"B"}}
    layers = topological_layers(groups, deps)
    assert layers == [["A"], ["B"], ["C"]]


def test_topological_layers_parallel() -> None:
    groups = {"A": ["s1"], "B": ["s2"], "C": ["s3"]}
    deps = {"B": {"A"}, "C": {"A"}}
    layers = topological_layers(groups, deps)
    assert layers == [["A"], ["B", "C"]]


def test_topological_layers_no_deps() -> None:
    groups = {"A": ["s1"], "B": ["s2"], "C": ["s3"]}
    deps = {}
    layers = topological_layers(groups, deps)
    assert layers == [["A", "B", "C"]]


def test_topological_layers_cycle_raises() -> None:
    groups = {"A": ["s1"], "B": ["s2"]}
    deps = {"A": {"B"}, "B": {"A"}}
    with pytest.raises(ValueError, match="cycle"):
        topological_layers(groups, deps)


def test_topological_layers_single_feature() -> None:
    groups = {"auth": ["s1", "s2"]}
    deps = {}
    layers = topological_layers(groups, deps)
    assert layers == [["auth"]]


def test_topological_layers_diamond() -> None:
    """A → B, A → C, B → D, C → D"""
    groups = {"A": ["s1"], "B": ["s2"], "C": ["s3"], "D": ["s4"]}
    deps = {"B": {"A"}, "C": {"A"}, "D": {"B", "C"}}
    layers = topological_layers(groups, deps)
    assert layers == [["A"], ["B", "C"], ["D"]]


# ── check_done tests ────────────────────────────────────────────────


def test_check_done_continues() -> None:
    state: OrchestratorState = {
        "execution_order": [["A"], ["B"]],
        "current_layer": 0,
    }
    assert check_done(state) == "execute_layer"


def test_check_done_finishes() -> None:
    state: OrchestratorState = {
        "execution_order": [["A"]],
        "current_layer": 1,
    }
    assert check_done(state) == "aggregate"


# ── aggregate_node tests ────────────────────────────────────────────


def test_aggregate_merges_results() -> None:
    state: OrchestratorState = {
        "completed_features": [
            FeatureResult(
                feature="auth",
                spec_ids=["s1"],
                status="success",
                artifacts=[{"id": "a1"}],
                tests=[{"id": "t1"}],
                error=None,
            ),
            FeatureResult(
                feature="profile",
                spec_ids=["s2"],
                status="success",
                artifacts=[{"id": "a2"}],
                tests=[{"id": "t2"}],
                error=None,
            ),
        ],
    }
    result = aggregate_node(state)
    assert len(result["all_artifacts"]) == 2
    assert len(result["all_tests"]) == 2


def test_aggregate_handles_empty() -> None:
    state: OrchestratorState = {"completed_features": []}
    result = aggregate_node(state)
    assert result["all_artifacts"] == []
    assert result["all_tests"] == []


# ── FeatureResult schema test ────────────────────────────────────────


def test_feature_result_schema() -> None:
    result = FeatureResult(
        feature="auth",
        spec_ids=["s1"],
        status="success",
        artifacts=[],
        tests=[],
        error=None,
    )
    assert set(result.keys()) == {"feature", "spec_ids", "status", "artifacts", "tests", "error"}
