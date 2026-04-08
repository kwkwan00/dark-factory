"""Tests for L4 capabilities: cross-feature learning, strategy adjustment, parallel execution."""

from __future__ import annotations

import threading

from dark_factory.agents.swarm import FeatureResult, MAX_HANDOFFS


# ── Thread-local isolation tests ─────────────────────────────────────


def test_thread_local_current_feature() -> None:
    """set_current_feature in one thread doesn't affect another."""
    import dark_factory.agents.tools as tools_mod

    results = {}

    def _set_and_read(name: str):
        tools_mod.set_current_feature(name)
        results[name] = tools_mod.get_current_feature()

    t1 = threading.Thread(target=_set_and_read, args=("feature-a",))
    t2 = threading.Thread(target=_set_and_read, args=("feature-b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["feature-a"] == "feature-a"
    assert results["feature-b"] == "feature-b"


def test_thread_local_recalled_ids() -> None:
    """recalled_memory_ids are isolated per thread."""
    import dark_factory.agents.tools as tools_mod

    results = {}

    def _add_and_read(thread_id: str, ids: list[str]):
        tools_mod.clear_recalled_memories()
        tools_mod.add_recalled_memory_ids(ids)
        results[thread_id] = list(tools_mod.get_recalled_memory_ids())

    t1 = threading.Thread(target=_add_and_read, args=("t1", ["a", "b"]))
    t2 = threading.Thread(target=_add_and_read, args=("t2", ["x", "y", "z"]))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["t1"] == ["a", "b"]
    assert results["t2"] == ["x", "y", "z"]


# ── Cross-feature learning tests ────────────────────────────────────


def test_run_feature_swarm_includes_run_context() -> None:
    """run_context is prepended to initial message."""
    from unittest.mock import MagicMock, patch

    with patch("dark_factory.agents.swarm.set_current_feature"):
        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": []}

        from dark_factory.agents.swarm import run_feature_swarm

        result = run_feature_swarm(
            mock_compiled, ["spec-1"], "auth",
            run_context="[PATTERN from payments] Always validate currency",
        )

        call_args = mock_compiled.invoke.call_args[0][0]
        msg = call_args["messages"][0]["content"]
        assert "Learnings from earlier features" in msg
        assert "Always validate currency" in msg
        assert result["status"] == "success"


def test_run_feature_swarm_respects_max_handoffs() -> None:
    """Custom max_handoffs appears in the initial message."""
    from unittest.mock import MagicMock, patch

    with patch("dark_factory.agents.swarm.set_current_feature"):
        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": []}

        from dark_factory.agents.swarm import run_feature_swarm

        run_feature_swarm(mock_compiled, ["spec-1"], "auth", max_handoffs=25)
        msg = mock_compiled.invoke.call_args[0][0]["messages"][0]["content"]
        assert "25 handoffs" in msg


# ── Strategy adjustment tests ────────────────────────────────────────


def test_adjust_strategy_triggers_on_low_pass_rate() -> None:
    from dark_factory.agents.orchestrator import make_adjust_strategy_node

    node = make_adjust_strategy_node(threshold=0.5)
    state = {
        "completed_features": [
            FeatureResult(feature="a", spec_ids=["s1"], status="error",
                         artifacts=[], tests=[], error="fail", eval_scores={}),
            FeatureResult(feature="b", spec_ids=["s2"], status="error",
                         artifacts=[], tests=[], error="fail", eval_scores={}),
        ],
        "execution_order": [["a", "b"]],
        "current_layer": 1,
        "strategy_overrides": {},
        "layer_pass_rates": [],
    }
    result = node(state)
    assert result["strategy_overrides"]["force_claude_agent"] is True
    assert result["strategy_overrides"]["max_handoffs"] == 30
    assert result["layer_pass_rates"] == [0.0]


def test_adjust_strategy_relaxes_on_recovery() -> None:
    from dark_factory.agents.orchestrator import make_adjust_strategy_node

    node = make_adjust_strategy_node(threshold=0.5)
    state = {
        "completed_features": [
            FeatureResult(feature="c", spec_ids=["s3"], status="success",
                         artifacts=[], tests=[], error=None, eval_scores={}),
        ],
        "execution_order": [["a", "b"], ["c"]],
        "current_layer": 2,
        "strategy_overrides": {"force_claude_agent": True, "max_handoffs": 30},
        "layer_pass_rates": [0.0],
    }
    result = node(state)
    assert "force_claude_agent" not in result["strategy_overrides"]
    assert "max_handoffs" not in result["strategy_overrides"]


def test_adjust_strategy_no_change_when_passing() -> None:
    from dark_factory.agents.orchestrator import make_adjust_strategy_node

    node = make_adjust_strategy_node(threshold=0.5)
    state = {
        "completed_features": [
            FeatureResult(feature="a", spec_ids=["s1"], status="success",
                         artifacts=[], tests=[], error=None, eval_scores={}),
        ],
        "execution_order": [["a"]],
        "current_layer": 1,
        "strategy_overrides": {},
        "layer_pass_rates": [],
    }
    result = node(state)
    assert result["strategy_overrides"] == {}


# ── Strategy suffix tests ────────────────────────────────────────────


def test_strategy_suffix_empty() -> None:
    from dark_factory.agents.swarm import _strategy_suffix

    assert _strategy_suffix(None) == ""
    assert _strategy_suffix({}) == ""


def test_strategy_suffix_force_claude() -> None:
    from dark_factory.agents.swarm import _strategy_suffix

    result = _strategy_suffix({"force_claude_agent": True, "max_handoffs": 30})
    assert "claude_agent_codegen" in result
    assert "30" in result


# ── Config tests ─────────────────────────────────────────────────────


def test_pipeline_config_max_parallel() -> None:
    from dark_factory.config import PipelineConfig

    assert PipelineConfig().max_parallel_features == 4


def test_evaluation_config_strategy_threshold() -> None:
    from dark_factory.config import EvaluationConfig

    assert EvaluationConfig().strategy_threshold == 0.5


# ── Orchestrator graph includes adjust_strategy ──────────────────────


def test_orchestrator_graph_has_adjust_strategy() -> None:
    """The orchestrator graph should include the adjust_strategy node."""
    from unittest.mock import MagicMock

    from dark_factory.agents.orchestrator import build_orchestrator
    from dark_factory.config import Settings

    settings = Settings()
    mock_repo = MagicMock()

    graph = build_orchestrator(settings, mock_repo)
    # StateGraph has nodes attribute
    node_names = set(graph.nodes.keys())
    assert "adjust_strategy" in node_names
    assert "execute_layer" in node_names
    assert "aggregate" in node_names
