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


def test_topological_layers_cycle_collapses_to_one_layer() -> None:
    """A 2-node cycle is collapsed into a single parallel layer so the
    pipeline can make progress. The old behaviour raised ``ValueError``
    and aborted the whole run whenever the decomposition planner
    produced circular capability-level deps. Now the cycle members
    run in parallel (they can't be ordered by definition) and the
    operator sees a ``dependency_cycles_collapsed`` log warning."""
    groups = {"A": ["s1"], "B": ["s2"]}
    deps = {"A": {"B"}, "B": {"A"}}
    layers = topological_layers(groups, deps)
    assert layers == [["A", "B"]]


def test_topological_layers_cycle_embedded_in_dag() -> None:
    """An SCC embedded in a larger DAG. The cycle (a, d, e) collapses
    into a single layer, then b and c — which both depend on d —
    follow in the next layer. Regression test for the capability-cycle
    bug observed in production (28 features caught in a cycle)."""
    groups = {
        "a": ["s1"],
        "b": ["s2"],
        "c": ["s3"],
        "d": ["s4"],
        "e": ["s5"],
    }
    deps = {
        "a": {"d"},  # a depends on d
        "b": {"d"},  # b depends on d
        "c": {"d"},  # c depends on d
        "d": {"e"},  # d depends on e
        "e": {"a"},  # e depends on a  ← closes the a→d→e→a cycle
    }
    layers = topological_layers(groups, deps)
    # The SCC {a, d, e} runs first as a single parallel layer.
    # Then b and c (which only depended on d) can run together.
    assert layers == [["a", "d", "e"], ["b", "c"]]


def test_topological_layers_self_loop_is_ignored() -> None:
    """A node that depends on itself is a pathological planner output
    (a spec "depends on itself"). Dropping the self-loop is the
    friendliest behaviour — it's never informative and would otherwise
    force the node into its own single-member SCC which has no impact
    on layering but clutters the cycles warning."""
    groups = {"A": ["s1"], "B": ["s2"]}
    deps = {"A": {"A"}, "B": {"A"}}
    layers = topological_layers(groups, deps)
    assert layers == [["A"], ["B"]]


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
        eval_scores={},
    )
    assert set(result.keys()) == {"feature", "spec_ids", "status", "artifacts", "tests", "error", "eval_scores"}
