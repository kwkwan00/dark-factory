"""Unit tests for ``AnthropicClient.complete_structured`` tool_use path.

Exercises the schema inliner and the tool_use branch of ``complete_structured``
with a mocked Anthropic SDK client. Never hits the real API.
"""

from __future__ import annotations



from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import anthropic

from dark_factory.llm.anthropic import (
    AnthropicClient,
    _build_tool_schema,
    _extract_tool_input,
    _inline_refs,
)
from dark_factory.models.domain import Scenario, Spec


# ── Schema inliner ──────────────────────────────────────────────────────────


def test_inline_refs_inlines_top_level_defs():
    schema = {
        "properties": {
            "child": {"$ref": "#/$defs/Child"},
        },
        "$defs": {
            "Child": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
    }
    out = _inline_refs(schema)
    assert "$defs" not in out
    assert out["properties"]["child"] == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }


def test_inline_refs_handles_definitions_key():
    schema = {
        "properties": {"x": {"$ref": "#/definitions/Thing"}},
        "definitions": {"Thing": {"type": "string"}},
    }
    out = _inline_refs(schema)
    assert out["properties"]["x"] == {"type": "string"}


def test_inline_refs_handles_nested_list_of_refs():
    schema = {
        "properties": {
            "items": {
                "type": "array",
                "items": {"$ref": "#/$defs/Item"},
            },
        },
        "$defs": {"Item": {"type": "object", "properties": {"k": {"type": "integer"}}}},
    }
    out = _inline_refs(schema)
    assert out["properties"]["items"]["items"] == {
        "type": "object",
        "properties": {"k": {"type": "integer"}},
    }


def test_inline_refs_breaks_cycles_with_permissive_object():
    schema = {
        "$ref": "#/$defs/Node",
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "child": {"$ref": "#/$defs/Node"},
                },
            },
        },
    }
    out = _inline_refs(schema)
    # The top-level $ref resolves to the Node definition; the self-ref inside
    # becomes a permissive object to avoid infinite recursion.
    assert out["type"] == "object"
    assert out["properties"]["child"] == {"type": "object"}


def test_build_tool_schema_for_spec_inlines_scenarios():
    """Spec has ``scenarios: list[Scenario]`` — the Scenario def should be
    inlined inside the array items rather than left as a $ref."""
    schema = _build_tool_schema(Spec)
    assert "$defs" not in schema
    scenarios_prop = schema["properties"]["scenarios"]
    assert scenarios_prop["type"] == "array"
    item_schema = scenarios_prop["items"]
    # Should be a plain object with Scenario's fields, not a $ref
    assert "$ref" not in item_schema
    assert item_schema["type"] == "object"
    assert set(item_schema["properties"].keys()) == {"name", "when", "then"}


# ── _extract_tool_input ─────────────────────────────────────────────────────


def test_extract_tool_input_returns_dict_from_tool_use_block():
    block = SimpleNamespace(type="tool_use", input={"id": "spec-1", "title": "X"})
    final = SimpleNamespace(content=[block])
    assert _extract_tool_input(final) == {"id": "spec-1", "title": "X"}


def test_extract_tool_input_ignores_non_tool_use_blocks():
    text_block = SimpleNamespace(type="text", text="hello")
    final = SimpleNamespace(content=[text_block])
    assert _extract_tool_input(final) is None


def test_extract_tool_input_picks_first_tool_use_when_multiple():
    first = SimpleNamespace(type="tool_use", input={"a": 1})
    second = SimpleNamespace(type="tool_use", input={"b": 2})
    final = SimpleNamespace(content=[first, second])
    assert _extract_tool_input(final) == {"a": 1}


# ── complete_structured happy path ──────────────────────────────────────────


def _mock_final_message(tool_input: dict, *, stop_reason: str = "tool_use"):
    """Build a mock final message shaped like an Anthropic SDK response."""
    usage = SimpleNamespace(
        input_tokens=120,
        output_tokens=450,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    block = SimpleNamespace(type="tool_use", input=tool_input)
    return SimpleNamespace(
        content=[block],
        stop_reason=stop_reason,
        usage=usage,
    )


def _mock_stream_context(final_message):
    """Return a context manager mimicking ``messages.stream(...)``."""
    stream = MagicMock()
    stream.get_final_message.return_value = final_message
    cm = MagicMock()
    cm.__enter__.return_value = stream
    cm.__exit__.return_value = False
    return cm


def test_complete_structured_returns_validated_spec():
    """End-to-end: mocked stream returns a tool_use block whose input
    matches the Spec schema — ``complete_structured`` should return a
    fully validated ``Spec`` instance."""
    tool_input = {
        "id": "spec-test-1",
        "title": "Test Spec",
        "description": "A test specification.",
        "requirement_ids": ["req-1"],
        "acceptance_criteria": ["must work"],
        "dependencies": [],
        "scenarios": [
            {"name": "happy path", "when": "input is valid", "then": "output is right"},
        ],
        "capability": "test-cap",
    }

    with patch("dark_factory.llm.anthropic.anthropic.Anthropic") as mock_cls:
        mock_sdk = MagicMock()
        mock_sdk.messages.stream.return_value = _mock_stream_context(
            _mock_final_message(tool_input)
        )
        mock_cls.return_value = mock_sdk

        client = AnthropicClient(api_key="sk-fake")
        result = client.complete_structured(
            prompt="Generate a spec.",
            response_model=Spec,
            system="You are an architect.",
        )

    assert isinstance(result, Spec)
    assert result.id == "spec-test-1"
    assert result.capability == "test-cap"
    assert len(result.scenarios) == 1
    assert result.scenarios[0] == Scenario(
        name="happy path", when="input is valid", then="output is right"
    )

    # Confirm the stream was called exactly once with tool_choice locked in
    assert mock_sdk.messages.stream.call_count == 1
    kwargs = mock_sdk.messages.stream.call_args.kwargs
    assert kwargs["tool_choice"] == {
        "type": "tool",
        "name": "return_structured_output",
    }
    assert len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["name"] == "return_structured_output"
    # Tool schema must be self-contained (no $defs leakage)
    assert "$defs" not in kwargs["tools"][0]["input_schema"]
    # H4 fix: timeout must be explicitly bounded (default 300s)
    assert kwargs["timeout"] == 300.0


def test_complete_structured_honours_custom_timeout():
    """H4 guard: caller can override the default timeout via
    ``timeout_seconds`` and the value threads through to the
    Anthropic SDK's ``timeout`` kwarg."""
    tool_input = {
        "id": "spec-test-1",
        "title": "Test Spec",
        "description": "A test.",
        "requirement_ids": ["req-1"],
        "acceptance_criteria": ["ok"],
        "dependencies": [],
        "scenarios": [{"name": "n", "when": "w", "then": "t"}],
        "capability": "cap",
    }

    with patch("dark_factory.llm.anthropic.anthropic.Anthropic") as mock_cls:
        mock_sdk = MagicMock()
        mock_sdk.messages.stream.return_value = _mock_stream_context(
            _mock_final_message(tool_input)
        )
        mock_cls.return_value = mock_sdk

        client = AnthropicClient(api_key="sk-fake")
        client.complete_structured(
            prompt="Generate a spec.",
            response_model=Spec,
            timeout_seconds=90.0,
        )

    kwargs = mock_sdk.messages.stream.call_args.kwargs
    assert kwargs["timeout"] == 90.0


def test_complete_non_structured_bounds_timeout():
    """H4 guard: the plain ``complete`` path also binds a timeout."""
    with patch("dark_factory.llm.anthropic.anthropic.Anthropic") as mock_cls:
        mock_sdk = MagicMock()

        class _FakeStream:
            text_stream: list = ["hello"]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get_final_message(self):
                return SimpleNamespace(
                    stop_reason="end_turn",
                    content=[],
                    usage=SimpleNamespace(
                        input_tokens=1,
                        output_tokens=1,
                        cache_read_input_tokens=None,
                        cache_creation_input_tokens=None,
                    ),
                )

        mock_sdk.messages.stream.return_value = _FakeStream()
        mock_cls.return_value = mock_sdk

        client = AnthropicClient(api_key="sk-fake")
        client.complete(prompt="hi", timeout_seconds=45.0)

    kwargs = mock_sdk.messages.stream.call_args.kwargs
    assert kwargs["timeout"] == 45.0


def _transient_error() -> anthropic.APIConnectionError:
    """Construct a genuine anthropic.APIConnectionError for retry tests.

    The retry loop classifies errors into transient (retry) vs permanent
    (bail). Only instances of APIConnectionError / APITimeoutError /
    InternalServerError / RateLimitError OR HTTP 5xx OR rate_limited=True
    count as transient. Plain RuntimeError is correctly considered
    permanent now that the classification is in place.
    """
    import httpx

    return anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )


def test_complete_structured_retries_on_transient_error_then_succeeds():
    """First stream call raises a transient APIConnectionError; second
    succeeds. Should retry exactly once."""
    tool_input = {
        "id": "spec-retry",
        "title": "Retry",
        "description": "ok",
        "requirement_ids": ["req-r"],
        "acceptance_criteria": ["c"],
        "dependencies": [],
        "scenarios": [],
        "capability": "retry",
    }

    with patch("dark_factory.llm.anthropic.anthropic.Anthropic") as mock_cls:
        mock_sdk = MagicMock()
        # First attempt: raise a transient error; second: return a valid stream.
        failing_cm = MagicMock()
        failing_cm.__enter__.side_effect = _transient_error()
        failing_cm.__exit__.return_value = False

        ok_cm = _mock_stream_context(_mock_final_message(tool_input))

        mock_sdk.messages.stream.side_effect = [failing_cm, ok_cm]
        mock_cls.return_value = mock_sdk

        client = AnthropicClient(api_key="sk-fake")
        result = client.complete_structured(
            prompt="Generate a spec.",
            response_model=Spec,
        )

    assert result.id == "spec-retry"
    assert mock_sdk.messages.stream.call_count == 2


def test_complete_structured_does_not_retry_on_permanent_error():
    """Auth / invalid-request errors and plain RuntimeError are classified
    as permanent — the retry loop should bail immediately on the first
    attempt rather than burning a second API call uselessly."""
    with patch("dark_factory.llm.anthropic.anthropic.Anthropic") as mock_cls:
        mock_sdk = MagicMock()
        failing_cm = MagicMock()
        failing_cm.__enter__.side_effect = RuntimeError("invalid request: bad schema")
        failing_cm.__exit__.return_value = False
        mock_sdk.messages.stream.return_value = failing_cm
        mock_cls.return_value = mock_sdk

        client = AnthropicClient(api_key="sk-fake")
        with pytest.raises(RuntimeError, match="invalid request"):
            client.complete_structured(prompt="x", response_model=Spec)

    # CRITICAL: permanent errors should be tried exactly ONCE.
    assert mock_sdk.messages.stream.call_count == 1


def test_complete_structured_retries_twice_on_repeated_transient():
    """Both attempts hit a transient error → raises the last one after
    exhausting the retry budget (2 attempts)."""
    with patch("dark_factory.llm.anthropic.anthropic.Anthropic") as mock_cls:
        mock_sdk = MagicMock()
        failing_cm = MagicMock()
        failing_cm.__enter__.side_effect = _transient_error()
        failing_cm.__exit__.return_value = False
        mock_sdk.messages.stream.return_value = failing_cm
        mock_cls.return_value = mock_sdk

        client = AnthropicClient(api_key="sk-fake")
        with pytest.raises(anthropic.APIConnectionError):
            client.complete_structured(prompt="x", response_model=Spec)

    assert mock_sdk.messages.stream.call_count == 2


def test_complete_structured_raises_when_no_tool_use_block_returned():
    """Defensive: if the model somehow replies with text instead of a
    tool_use block, we raise a clear error rather than silently swallowing it."""
    text_block = SimpleNamespace(type="text", text="Sorry, I can't do that.")
    final = SimpleNamespace(
        content=[text_block],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=50,
            output_tokens=10,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )

    with patch("dark_factory.llm.anthropic.anthropic.Anthropic") as mock_cls:
        mock_sdk = MagicMock()
        mock_sdk.messages.stream.return_value = _mock_stream_context(final)
        mock_cls.return_value = mock_sdk

        client = AnthropicClient(api_key="sk-fake")
        with pytest.raises(RuntimeError, match="no tool_use block"):
            client.complete_structured(prompt="x", response_model=Spec)
