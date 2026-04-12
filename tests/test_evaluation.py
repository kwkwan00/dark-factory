"""Tests for the AI evaluation module."""

from __future__ import annotations

from unittest.mock import patch

from dark_factory.evaluation.metrics import (
    build_spec_test_case,
    build_test_case,
    get_eval_model,
)


def test_eval_model_defaults_to_gpt54() -> None:
    """get_eval_model() defaults to gpt-5.4 but can be overridden via
    the EVAL_MODEL env var (read at module import time) or via
    ``set_eval_model`` / the Settings-tab PATCH at runtime."""
    assert get_eval_model()  # non-empty
    import os
    if not os.getenv("EVAL_MODEL"):
        assert get_eval_model() == "gpt-5.4"


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
    """Verify metric builders pass the configured eval model (dynamic)."""
    with patch("dark_factory.evaluation.metrics.GEval") as mock_geval:
        from dark_factory.evaluation.metrics import (
            build_correctness_metric,
            get_eval_model,
        )

        build_correctness_metric()
        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["name"] == "Test Correctness"
        # Builder must call get_eval_model() at construction time so
        # runtime Settings-tab changes propagate.
        assert call_kwargs["model"] == get_eval_model()


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
    """Verify spec metric builders pass the configured eval model (dynamic)."""
    with patch("dark_factory.evaluation.metrics.GEval") as mock_geval:
        from dark_factory.evaluation.metrics import (
            build_spec_correctness_metric,
            get_eval_model,
        )

        build_spec_correctness_metric()
        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["name"] == "Spec Correctness"
        assert call_kwargs["model"] == get_eval_model()


def test_set_eval_model_propagates_to_builders() -> None:
    """Changing the eval model at runtime must affect the NEXT builder
    call. Regression test for the ``set_eval_model`` refactor — the
    previous ``EVAL_MODEL`` constant was captured at import time and
    never picked up Settings-tab PATCHes."""
    with patch("dark_factory.evaluation.metrics.GEval") as mock_geval:
        from dark_factory.evaluation.metrics import (
            build_spec_correctness_metric,
            get_eval_model,
            set_eval_model,
        )

        original = get_eval_model()
        try:
            set_eval_model("gpt-test-override-1234")
            build_spec_correctness_metric()
            assert mock_geval.call_args.kwargs["model"] == "gpt-test-override-1234"
        finally:
            set_eval_model(original)


def test_set_eval_model_rejects_empty_string() -> None:
    """Defensive validator: empty strings would silently break GEval."""
    import pytest as _pytest

    from dark_factory.evaluation.metrics import set_eval_model

    with _pytest.raises(ValueError, match="non-empty string"):
        set_eval_model("")


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
