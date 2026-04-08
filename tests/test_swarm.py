"""Tests for the swarm harness."""

from __future__ import annotations

from dark_factory.agents.swarm import MAX_HANDOFFS


def test_max_handoffs_constant() -> None:
    assert MAX_HANDOFFS == 50


def test_build_planner_has_handoff_tool() -> None:
    """Planner agent should include a transfer_to_coder handoff tool."""
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_planner

        _build_planner("anthropic:claude-sonnet-4-6")

        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools") or call_kwargs[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert any("coder" in n for n in tool_names), f"Expected handoff to coder in {tool_names}"


def test_build_coder_has_claude_agent_and_handoff() -> None:
    """Coder agent should include claude_agent_codegen and transfer_to_reviewer."""
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_coder

        _build_coder("anthropic:claude-sonnet-4-6")

        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools") or call_kwargs[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "claude_agent_codegen" in tool_names, f"Expected claude_agent_codegen in {tool_names}"
        assert any("reviewer" in n for n in tool_names), f"Expected handoff to reviewer in {tool_names}"


def test_build_reviewer_has_both_handoffs() -> None:
    """Reviewer agent should have handoffs to both coder (reject) and tester (approve)."""
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_reviewer

        _build_reviewer("anthropic:claude-sonnet-4-6")

        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools") or call_kwargs[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert any("coder" in n for n in tool_names), f"Expected handoff to coder in {tool_names}"
        assert any("tester" in n for n in tool_names), f"Expected handoff to tester in {tool_names}"


def test_build_tester_has_handoff_to_planner() -> None:
    """Tester agent should include a transfer_to_planner handoff tool."""
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_tester

        _build_tester("anthropic:claude-sonnet-4-6")

        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools") or call_kwargs[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert any("planner" in n for n in tool_names), f"Expected handoff to planner in {tool_names}"
