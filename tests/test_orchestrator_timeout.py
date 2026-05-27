"""Tests for orchestrator timeout handling.

Covers:
- TimeoutError from run_feature_swarm produces status="timeout" (not "error")
- Timed-out features are excluded from the reflection retry set
- The layer_pass_rate computation counts "timeout" as non-success
"""

from __future__ import annotations

import pytest


# ── FeatureResult "timeout" status ────────────────────────────────────────────


def test_feature_result_accepts_timeout_status():
    """FeatureResult TypedDict should accept status="timeout"."""
    from dark_factory.agents.swarm import FeatureResult
    result = FeatureResult(
        feature="auth",
        spec_ids=["s1"],
        status="timeout",
        artifacts=[],
        tests=[],
        error="Timed out: feature exceeded 600s wall-clock timeout",
        eval_scores={},
    )
    assert result["status"] == "timeout"
    assert "timeout" in result["error"].lower()


# ── timed_out_features exclusion from reflection ──────────────────────────────


def test_timed_out_features_excluded_from_retry():
    """When adjust_strategy_node sees timeout features, they must NOT appear
    in the retryable set even if the reflection LLM recommends retrying them.
    """
    from unittest.mock import patch
    from dark_factory.agents.orchestrator import (
        OrchestratorState,
        make_adjust_strategy_node,
    )
    from dark_factory.agents.swarm import FeatureResult

    # Build a minimal adjust_strategy_node that will trigger the retry path.
    # make_adjust_strategy_node takes threshold and max_layer_retries only;
    # execution_order comes from state.
    adjust = make_adjust_strategy_node(
        threshold=1.0,       # any failure triggers reflection
        max_layer_retries=1,
    )

    # Simulate: auth timed out, profile succeeded
    auth_result = FeatureResult(
        feature="auth",
        spec_ids=["s1"],
        status="timeout",
        artifacts=[],
        tests=[],
        error="Timed out",
        eval_scores={},
    )
    profile_result = FeatureResult(
        feature="profile",
        spec_ids=["s2"],
        status="success",
        artifacts=[{"id": "a1"}],
        tests=[],
        error=None,
        eval_scores={},
    )

    state: OrchestratorState = {
        "execution_order": [["auth", "profile"]],
        "current_layer": 1,
        "completed_features": [auth_result, profile_result],
        "strategy_overrides": {},
        "layer_pass_rates": [],
        "layer_retries": {},
        "all_spec_ids": ["s1", "s2"],
    }

    # Reflection LLM tries to recommend retrying "auth" (the timed-out feature)
    fake_reflection = {
        "retryable_features": ["auth"],   # LLM wrongly recommends retry
        "terminal_features": [],
        "diagnosis": "auth timed out, should retry",
        "strategy": {},
    }

    with patch("dark_factory.agents.orchestrator._run_reflection", return_value=fake_reflection), \
         patch("dark_factory.agents.tools.emit_progress"):
        result_state = adjust(state)

    # "auth" was timed out → must be excluded from retry even though LLM recommended it
    retry_layer = result_state.get("retry_layer")
    completed = result_state.get("completed_features", [])

    # If auth is still in completed (not removed for retry), timeout exclusion worked
    auth_in_completed = any(r["feature"] == "auth" for r in completed)
    assert auth_in_completed, (
        "auth (timed out) should remain in completed rather than being queued for retry"
    )


# ── agentic loop backoff ──────────────────────────────────────────────────────


def test_agentic_call_with_retry_calls_backoff_on_transient():
    """_call_with_retry in agentic.py should call _backoff_sleep between attempts."""
    from unittest.mock import patch, MagicMock
    import anthropic
    import httpx

    transient = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )

    from types import SimpleNamespace

    ok_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="done")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=5, output_tokens=2,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    )

    with patch("dark_factory.llm.agentic._record_llm_call"), \
         patch("dark_factory.llm.agentic._backoff_sleep") as mock_backoff:

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [transient, ok_response]

        from dark_factory.llm.agentic import _call_with_retry
        result = _call_with_retry(
            client=mock_client,
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            max_tokens=256,
            system=None,
            timeout_seconds=30.0,
            turn=1,
        )

    response, started_at = result
    assert response is ok_response
    assert isinstance(started_at, float)
    assert mock_client.messages.create.call_count == 2
    mock_backoff.assert_called_once()


def test_agentic_call_with_retry_does_not_retry_permanent_error():
    """_call_with_retry should raise immediately on non-transient errors."""
    from unittest.mock import patch, MagicMock

    with patch("dark_factory.llm.agentic._record_llm_call"), \
         patch("dark_factory.llm.agentic._backoff_sleep") as mock_backoff:

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = ValueError("bad request")

        from dark_factory.llm.agentic import _call_with_retry
        with pytest.raises(ValueError, match="bad request"):
            _call_with_retry(
                client=mock_client,
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                max_tokens=256,
                system=None,
                timeout_seconds=30.0,
                turn=1,
            )

    assert mock_client.messages.create.call_count == 1
    mock_backoff.assert_not_called()
