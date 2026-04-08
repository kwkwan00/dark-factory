"""Tests for the AI evaluation module."""

from __future__ import annotations

from unittest.mock import patch

from dark_factory.evaluation.metrics import (
    EVAL_MODEL,
    build_spec_test_case,
    build_test_case,
)


def test_eval_model_is_gpt54() -> None:
    assert EVAL_MODEL == "gpt-5.4"


def test_build_test_case() -> None:
    tc = build_test_case(
        spec_title="User Auth",
        acceptance_criteria=["Login returns JWT", "Invalid password returns 401"],
        source_code="def login(user, pw): ...",
        test_code="def test_login(): assert login('a', 'b')",
    )
    assert "User Auth" in tc.input
    assert "Login returns JWT" in tc.input
    assert "def test_login" in tc.actual_output
    assert "Login returns JWT" in tc.expected_output
    assert "def login" in tc.context[0]


def test_build_metrics_use_openai_model() -> None:
    """Verify metric builders pass gpt-5.4 as the model."""
    with patch("dark_factory.evaluation.metrics.GEval") as mock_geval:
        from dark_factory.evaluation.metrics import build_correctness_metric

        build_correctness_metric()
        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["name"] == "Test Correctness"
        assert call_kwargs["model"] == "gpt-5.4"


def test_build_spec_test_case() -> None:
    tc = build_spec_test_case(
        requirement_title="User Authentication",
        requirement_description="Users must log in with email and password.",
        spec_json='{"id": "spec-1", "title": "Auth Module"}',
    )
    assert "User Authentication" in tc.input
    assert "spec-1" in tc.actual_output
    assert "email and password" in tc.expected_output


def test_build_spec_metrics_use_openai() -> None:
    """Verify spec metric builders pass gpt-5.4."""
    with patch("dark_factory.evaluation.metrics.GEval") as mock_geval:
        from dark_factory.evaluation.metrics import build_spec_correctness_metric

        build_spec_correctness_metric()
        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["name"] == "Spec Correctness"
        assert call_kwargs["model"] == "gpt-5.4"


def test_evaluate_spec_tool_exists() -> None:
    from dark_factory.agents.tools import EVAL_TOOLS, evaluate_spec

    assert evaluate_spec in EVAL_TOOLS


def test_planner_agent_has_evaluate_spec() -> None:
    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_planner

        _build_planner("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "evaluate_spec" in tool_names


def test_evaluate_tests_tool_exists() -> None:
    from dark_factory.agents.tools import EVAL_TOOLS, evaluate_tests

    assert evaluate_tests in EVAL_TOOLS


def test_tester_agent_has_eval_tool() -> None:
    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_tester

        _build_tester("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "evaluate_tests" in tool_names


def test_reviewer_agent_has_eval_tool() -> None:
    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock_agent"
        from dark_factory.agents.swarm import _build_reviewer

        _build_reviewer("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "evaluate_tests" in tool_names
