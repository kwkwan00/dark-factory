"""Tests for ``dark_factory.llm.agentic`` — multi-turn tool-use loop."""



from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, input_dict: dict, tool_id: str = "tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_dict, id=tool_id)


def _make_response(content, stop_reason="end_turn", input_tokens=10, output_tokens=5):
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)


# ── Tests ─────────────────────────────────────────────────────────────────────


@patch("dark_factory.llm.agentic._record_llm_call")
@patch("dark_factory.llm.agentic.anthropic")
def test_single_turn_end_turn(mock_anthropic, mock_record, tmp_path: Path):
    """Model responds with text and end_turn on the first call."""
    client = MagicMock()
    mock_anthropic.Anthropic.return_value = client
    client.messages.create.return_value = _make_response(
        [_text_block("Hello world")], "end_turn"
    )

    from dark_factory.llm.agentic import run_agentic_loop

    result = run_agentic_loop(
        prompt="Say hello",
        allowed_tools=["Read"],
        sandbox_root=tmp_path,
        max_turns=5,
        model="claude-sonnet-4-6",
    )
    assert result == "Hello world"
    assert client.messages.create.call_count == 1


@patch("dark_factory.llm.agentic._record_llm_call")
@patch("dark_factory.llm.agentic.anthropic")
def test_tool_use_then_end_turn(mock_anthropic, mock_record, tmp_path: Path):
    """Model calls a tool, gets result, then finishes."""
    # Create a file for the Read tool to find
    (tmp_path / "data.txt").write_text("file contents")

    client = MagicMock()
    mock_anthropic.Anthropic.return_value = client

    # Turn 1: model requests Read tool
    turn1 = _make_response(
        [_tool_use_block("Read", {"file_path": "data.txt"})],
        "tool_use",
    )
    # Turn 2: model responds with text
    turn2 = _make_response([_text_block("I read the file")], "end_turn")
    client.messages.create.side_effect = [turn1, turn2]

    from dark_factory.llm.agentic import run_agentic_loop

    result = run_agentic_loop(
        prompt="Read data.txt",
        allowed_tools=["Read"],
        sandbox_root=tmp_path,
        max_turns=5,
        model="claude-sonnet-4-6",
    )
    assert "I read the file" in result
    assert client.messages.create.call_count == 2


@patch("dark_factory.llm.agentic._record_llm_call")
@patch("dark_factory.llm.agentic.anthropic")
def test_max_turns_exhausted(mock_anthropic, mock_record, tmp_path: Path):
    """Loop should stop and return partial text when max_turns is reached."""
    client = MagicMock()
    mock_anthropic.Anthropic.return_value = client

    # Every turn requests another tool call (infinite loop)
    tool_response = _make_response(
        [_text_block("thinking..."), _tool_use_block("Read", {"file_path": "x.txt"})],
        "tool_use",
    )
    (tmp_path / "x.txt").write_text("x")
    client.messages.create.return_value = tool_response

    from dark_factory.llm.agentic import run_agentic_loop

    result = run_agentic_loop(
        prompt="Loop forever",
        allowed_tools=["Read"],
        sandbox_root=tmp_path,
        max_turns=3,
        model="claude-sonnet-4-6",
    )
    assert "turn budget" in result
    assert client.messages.create.call_count == 3


@patch("dark_factory.llm.agentic._record_llm_call")
@patch("dark_factory.llm.agentic.anthropic")
def test_max_tokens_returns_partial(mock_anthropic, mock_record, tmp_path: Path):
    """stop_reason=max_tokens should return partial text."""
    client = MagicMock()
    mock_anthropic.Anthropic.return_value = client
    client.messages.create.return_value = _make_response(
        [_text_block("partial output")], "max_tokens"
    )

    from dark_factory.llm.agentic import run_agentic_loop

    result = run_agentic_loop(
        prompt="Write a lot",
        allowed_tools=[],
        sandbox_root=tmp_path,
        max_turns=5,
        model="claude-sonnet-4-6",
    )
    assert result == "partial output"


@patch("dark_factory.llm.agentic._record_llm_call")
@patch("dark_factory.llm.agentic.anthropic")
def test_tool_error_continues_loop(mock_anthropic, mock_record, tmp_path: Path):
    """A tool execution error should produce is_error=True result and continue."""
    client = MagicMock()
    mock_anthropic.Anthropic.return_value = client

    # Turn 1: model tries to read a missing file
    turn1 = _make_response(
        [_tool_use_block("Read", {"file_path": "missing.txt"})],
        "tool_use",
    )
    # Turn 2: model says done
    turn2 = _make_response([_text_block("OK, file not found")], "end_turn")
    client.messages.create.side_effect = [turn1, turn2]

    from dark_factory.llm.agentic import run_agentic_loop

    result = run_agentic_loop(
        prompt="Try reading a missing file",
        allowed_tools=["Read"],
        sandbox_root=tmp_path,
        max_turns=5,
        model="claude-sonnet-4-6",
    )
    assert "OK, file not found" in result

    # Check that the tool_result was sent with is_error=True
    second_call_messages = client.messages.create.call_args_list[1][1]["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["is_error"] is True


@patch("dark_factory.llm.agentic._record_llm_call")
@patch("dark_factory.llm.agentic.anthropic")
def test_write_tool_creates_file(mock_anthropic, mock_record, tmp_path: Path):
    """Write tool should actually create a file on disk."""
    client = MagicMock()
    mock_anthropic.Anthropic.return_value = client

    turn1 = _make_response(
        [_tool_use_block("Write", {"file_path": "out.py", "content": "print('hi')"})],
        "tool_use",
    )
    turn2 = _make_response([_text_block("Done")], "end_turn")
    client.messages.create.side_effect = [turn1, turn2]

    from dark_factory.llm.agentic import run_agentic_loop

    run_agentic_loop(
        prompt="Write a file",
        allowed_tools=["Write"],
        sandbox_root=tmp_path,
        max_turns=5,
        model="claude-sonnet-4-6",
    )
    assert (tmp_path / "out.py").read_text() == "print('hi')"
