"""Tests for L4 eval-memory feedback loop, adaptive thresholds, and run tracking."""

from __future__ import annotations

from dark_factory.config import EvaluationConfig, Settings
from dark_factory.evaluation.adaptive import compute_adaptive_threshold
from dark_factory.memory.schema import MEMORY_SCHEMA_STATEMENTS


# ── Config tests ─────────────────────────────────────────────────────


def test_evaluation_config_defaults() -> None:
    config = EvaluationConfig()
    assert config.base_threshold == 0.5
    assert config.adaptive is True
    assert config.decay_factor == 0.95
    assert config.boost_delta == 0.1
    assert config.demote_delta == 0.05
    assert config.trend_window == 5
    assert config.threshold_min == 0.3
    assert config.threshold_max == 0.9


def test_evaluation_config_in_settings() -> None:
    settings = Settings()
    assert settings.evaluation.base_threshold == 0.5
    assert settings.evaluation.adaptive is True


# ── Adaptive threshold tests ─────────────────────────────────────────


def test_adaptive_insufficient_data() -> None:
    result = compute_adaptive_threshold(
        base_threshold=0.5, recent_scores=[0.6, 0.7],
    )
    assert result == 0.5


def test_adaptive_trending_up() -> None:
    # First half: [0.3, 0.4], second half: [0.6, 0.7, 0.8]
    result = compute_adaptive_threshold(
        base_threshold=0.5, recent_scores=[0.3, 0.4, 0.6, 0.7, 0.8],
    )
    assert result == 0.55  # 0.5 + 0.05


def test_adaptive_trending_down() -> None:
    result = compute_adaptive_threshold(
        base_threshold=0.5, recent_scores=[0.8, 0.7, 0.4, 0.3, 0.2],
    )
    assert result == 0.45  # 0.5 - 0.05


def test_adaptive_flat() -> None:
    result = compute_adaptive_threshold(
        base_threshold=0.5, recent_scores=[0.5, 0.5, 0.5, 0.5, 0.5],
    )
    assert result == 0.5


def test_adaptive_clamped_max() -> None:
    result = compute_adaptive_threshold(
        base_threshold=0.88,
        recent_scores=[0.3, 0.4, 0.7, 0.8, 0.9],
        threshold_max=0.9,
    )
    assert result == 0.9


def test_adaptive_clamped_min() -> None:
    result = compute_adaptive_threshold(
        base_threshold=0.32,
        recent_scores=[0.8, 0.7, 0.3, 0.2, 0.1],
        threshold_min=0.3,
    )
    assert result == 0.3


# ── Schema tests ─────────────────────────────────────────────────────


def test_schema_has_eval_result_constraint() -> None:
    assert any("EvalResult" in s for s in MEMORY_SCHEMA_STATEMENTS)


def test_schema_has_run_constraint() -> None:
    assert any("run_id" in s and "Run" in s for s in MEMORY_SCHEMA_STATEMENTS)


def test_schema_has_eval_result_indexes() -> None:
    assert any("eval_result_spec" in s for s in MEMORY_SCHEMA_STATEMENTS)
    assert any("eval_result_run" in s for s in MEMORY_SCHEMA_STATEMENTS)
    assert any("eval_result_type" in s for s in MEMORY_SCHEMA_STATEMENTS)


# ── Tool state tests ────────────────────────────────────────────────


def test_recall_memories_tracks_ids() -> None:
    """recall_memories should populate _recalled_memory_ids."""
    import dark_factory.agents.tools as tools_mod
    from unittest.mock import MagicMock

    original_repo = tools_mod._memory_repo

    mock_repo = MagicMock()
    mock_repo.get_related_memories.return_value = [
        {"id": "pattern-abc", "description": "test"},
        {"id": "mistake-def", "description": "test2"},
    ]
    tools_mod._memory_repo = mock_repo
    tools_mod.clear_recalled_memories()

    try:
        tools_mod.recall_memories.invoke({"feature_name": "auth"})
        ids = tools_mod.get_recalled_memory_ids()
        assert "pattern-abc" in ids
        assert "mistake-def" in ids
    finally:
        tools_mod._memory_repo = original_repo
        tools_mod.clear_recalled_memories()


def test_eval_history_tools_exist() -> None:
    from dark_factory.agents.tools import (
        EVAL_HISTORY_TOOLS,
        query_eval_history,
        query_run_history,
    )
    assert query_eval_history in EVAL_HISTORY_TOOLS
    assert query_run_history in EVAL_HISTORY_TOOLS


def test_eval_tools_disabled_without_run_id() -> None:
    """Auto-persist should be skipped when no run_id is set."""
    import dark_factory.agents.tools as tools_mod

    original_run_id = tools_mod._current_run_id
    tools_mod._current_run_id = ""
    try:
        # _auto_persist_eval should be a no-op
        tools_mod._auto_persist_eval({}, spec_id="test", eval_type="spec")
        # No error means it handled gracefully
    finally:
        tools_mod._current_run_id = original_run_id


# ── Aggregate scoring tests ─────────────────────────────────────────


def test_aggregate_computes_pass_rate() -> None:
    from dark_factory.agents.orchestrator import aggregate_node
    from dark_factory.agents.swarm import FeatureResult

    state = {
        "completed_features": [
            FeatureResult(feature="a", spec_ids=["s1"], status="success",
                         artifacts=[], tests=[], error=None, eval_scores={}),
            FeatureResult(feature="b", spec_ids=["s2"], status="error",
                         artifacts=[], tests=[], error="fail", eval_scores={}),
        ],
    }
    result = aggregate_node(state)
    assert result["pass_rate"] == 0.5


def test_aggregate_computes_worst_features() -> None:
    from dark_factory.agents.orchestrator import aggregate_node
    from dark_factory.agents.swarm import FeatureResult

    state = {
        "completed_features": [
            FeatureResult(feature="good", spec_ids=["s1"], status="success",
                         artifacts=[], tests=[], error=None, eval_scores={}),
            FeatureResult(feature="bad", spec_ids=["s2"], status="error",
                         artifacts=[], tests=[], error="crashed", eval_scores={}),
        ],
    }
    result = aggregate_node(state)
    worst = result["worst_features"]
    assert any(w["feature"] == "bad" for w in worst)


# ── FeatureResult tests ─────────────────────────────────────────────


def test_feature_result_has_eval_scores() -> None:
    from dark_factory.agents.swarm import FeatureResult

    result = FeatureResult(
        feature="auth", spec_ids=["s1"], status="success",
        artifacts=[], tests=[], error=None, eval_scores={"s1": {"score": 0.8}},
    )
    assert "eval_scores" in result
    assert result["eval_scores"]["s1"]["score"] == 0.8


# ── Planner/Reviewer have history tools ──────────────────────────────


def test_planner_has_eval_history_tools() -> None:
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock"
        from dark_factory.agents.swarm import _build_planner
        _build_planner("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        names = [getattr(t, "name", str(t)) for t in tools]
        assert "query_eval_history" in names
        assert "query_run_history" in names


def test_reviewer_has_eval_history_tools() -> None:
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock"
        from dark_factory.agents.swarm import _build_reviewer
        _build_reviewer("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        names = [getattr(t, "name", str(t)) for t in tools]
        assert "query_eval_history" in names
        assert "query_run_history" in names
