"""Unit tests for LangChainClient retry logic and backoff.

Exercises complete() and complete_structured() retry paths with mocked
LangChain invocations. Never hits the real API.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_client():
    from dark_factory.llm.langchain import LangChainClient
    with patch("dark_factory.llm.langchain.ChatAnthropic"):
        return LangChainClient(api_key="sk-fake", model="claude-sonnet-4-6")


def _transient_exc():
    """Return a transient APIConnectionError that _is_transient() classifies as retryable."""
    import anthropic
    import httpx
    return anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )


def _ok_response(text: str = "result"):
    return SimpleNamespace(
        content=text,
        usage_metadata={"input_tokens": 5, "output_tokens": 3},
    )


# ── complete() tests ──────────────────────────────────────────────────────────


def test_complete_retries_once_on_transient():
    """complete() retries once when llm.invoke() raises a transient error."""
    client = _make_client()
    client.llm = MagicMock()
    client.llm.invoke.side_effect = [_transient_exc(), _ok_response("hello")]

    with patch("dark_factory.llm.langchain._backoff_sleep") as mock_backoff:
        result = client.complete(prompt="hi")

    assert result == "hello"
    assert client.llm.invoke.call_count == 2
    mock_backoff.assert_called_once()


def test_complete_does_not_retry_on_permanent_error():
    """complete() bails immediately on a non-transient error."""
    client = _make_client()
    client.llm = MagicMock()
    client.llm.invoke.side_effect = ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        client.complete(prompt="hi")

    assert client.llm.invoke.call_count == 1


def test_complete_raises_after_exhausting_retries():
    """Both attempts fail with a transient error → last error re-raised."""
    client = _make_client()
    client.llm = MagicMock()
    client.llm.invoke.side_effect = [_transient_exc(), _transient_exc()]

    with patch("dark_factory.llm.langchain._backoff_sleep"):
        import anthropic
        with pytest.raises(anthropic.APIConnectionError):
            client.complete(prompt="hi")

    assert client.llm.invoke.call_count == 2


def test_complete_succeeds_first_attempt_no_backoff():
    """When the first attempt succeeds, _backoff_sleep is never called."""
    client = _make_client()
    client.llm = MagicMock()
    client.llm.invoke.return_value = _ok_response("ok")

    with patch("dark_factory.llm.langchain._backoff_sleep") as mock_backoff:
        result = client.complete(prompt="hi")

    assert result == "ok"
    mock_backoff.assert_not_called()
    assert client.llm.invoke.call_count == 1


# ── complete_structured() tests ───────────────────────────────────────────────


def test_complete_structured_retries_on_transient():
    """complete_structured() retries once on transient errors with backoff."""
    from pydantic import BaseModel

    class Out(BaseModel):
        value: str

    client = _make_client()
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = [_transient_exc(), Out(value="x")]
    client.llm = MagicMock()
    client.llm.with_structured_output.return_value = structured_llm

    with patch("dark_factory.llm.langchain._backoff_sleep") as mock_backoff:
        result = client.complete_structured(prompt="hi", response_model=Out)

    assert result.value == "x"
    assert structured_llm.invoke.call_count == 2
    mock_backoff.assert_called_once()


def test_complete_structured_does_not_retry_on_permanent_error():
    """complete_structured() bails on permanent (non-transient) errors."""
    from pydantic import BaseModel

    class Out(BaseModel):
        value: str

    client = _make_client()
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = ValueError("bad schema")
    client.llm = MagicMock()
    client.llm.with_structured_output.return_value = structured_llm

    with pytest.raises(ValueError, match="bad schema"):
        client.complete_structured(prompt="hi", response_model=Out)

    assert structured_llm.invoke.call_count == 1


def test_complete_structured_raises_after_both_transient():
    """Both attempts fail → last error re-raised after retry budget exhausted."""
    from pydantic import BaseModel
    import anthropic

    class Out(BaseModel):
        value: str

    client = _make_client()
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = [_transient_exc(), _transient_exc()]
    client.llm = MagicMock()
    client.llm.with_structured_output.return_value = structured_llm

    with patch("dark_factory.llm.langchain._backoff_sleep"):
        with pytest.raises(anthropic.APIConnectionError):
            client.complete_structured(prompt="hi", response_model=Out)

    assert structured_llm.invoke.call_count == 2
