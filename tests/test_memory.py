"""Tests for the procedural memory system."""

from __future__ import annotations

from dark_factory.config import MemoryConfig, Settings
from dark_factory.memory.schema import MEMORY_SCHEMA_STATEMENTS


def test_memory_config_defaults() -> None:
    config = MemoryConfig()
    assert config.database == "memory"
    assert config.enabled is True


def test_memory_config_in_settings() -> None:
    settings = Settings()
    assert settings.memory.database == "memory"
    assert settings.memory.enabled is True


def test_memory_schema_has_constraints() -> None:
    labels = ["Pattern", "Mistake", "Solution", "Strategy"]
    for label in labels:
        assert any(label in stmt for stmt in MEMORY_SCHEMA_STATEMENTS), (
            f"Missing constraint for {label}"
        )


def test_memory_schema_has_indexes() -> None:
    assert any("pattern_agent" in stmt for stmt in MEMORY_SCHEMA_STATEMENTS)
    assert any("mistake_error_type" in stmt for stmt in MEMORY_SCHEMA_STATEMENTS)
    assert any("strategy_agent" in stmt for stmt in MEMORY_SCHEMA_STATEMENTS)


def test_memory_tools_disabled_returns_message() -> None:
    """When _memory_repo is None, tools return a disabled message."""
    import dark_factory.agents.tools as tools_mod

    # Ensure memory is disabled
    original = tools_mod._memory_repo
    tools_mod._memory_repo = None
    try:
        assert tools_mod.recall_memories.invoke({"feature_name": "auth"}) == "Memory system is disabled."
        assert tools_mod.search_memory.invoke({"keywords": "test"}) == "Memory system is disabled."
        assert tools_mod.record_pattern.invoke({"description": "x", "context": "y"}) == "Memory system is disabled."
        assert tools_mod.record_mistake.invoke({"description": "x", "error_type": "y", "trigger_context": "z"}) == "Memory system is disabled."
        assert tools_mod.record_solution.invoke({"description": "x"}) == "Memory system is disabled."
        assert tools_mod.record_strategy.invoke({"description": "x", "applicability": "y"}) == "Memory system is disabled."
    finally:
        tools_mod._memory_repo = original


def test_swarm_planner_has_memory_tools() -> None:
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_planner

        _build_planner("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "recall_memories" in tool_names
        assert "record_strategy" in tool_names


def test_swarm_coder_has_memory_tools() -> None:
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_coder

        _build_coder("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "recall_memories" in tool_names
        assert "record_pattern" in tool_names


def test_swarm_reviewer_has_memory_tools() -> None:
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_reviewer

        _build_reviewer("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "recall_memories" in tool_names
        assert "record_mistake" in tool_names
        assert "record_solution" in tool_names


def test_swarm_tester_has_memory_tools() -> None:
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_tester

        _build_tester("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "recall_memories" in tool_names
        assert "record_mistake" in tool_names
