"""Tests for fine-grained swarm progress events: tool_call, tool_result,
agent_handoff, and agent_active emitted from langgraph stream chunks."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from dark_factory.agents.progress import ProgressBroker
from dark_factory.agents.swarm import (
    _emit_message_events,
    _get_tool_calls,
    _is_tool_message,
    run_feature_swarm,
)
from dark_factory.agents.tools import set_progress_broker


# ── Helpers ───────────────────────────────────────────────────────────────────


def _drain(broker: ProgressBroker) -> list[dict]:
    """Drain all events from a fresh subscription. Runs on a tiny event loop."""
    async def _run():
        q = broker.subscribe(include_history=True)
        await asyncio.sleep(0.01)
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        return events
    return asyncio.run(_run())


# ── _get_tool_calls / _is_tool_message ────────────────────────────────────────


def test_get_tool_calls_dict_format():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "write_file", "args": {"path": "x.py"}, "id": "tc1"}],
    )
    calls = _get_tool_calls(msg)
    assert len(calls) == 1
    assert calls[0]["name"] == "write_file"
    assert calls[0]["args"] == {"path": "x.py"}


def test_get_tool_calls_empty_message():
    msg = AIMessage(content="just text, no tools")
    assert _get_tool_calls(msg) == []


def test_is_tool_message_for_tool_message():
    msg = ToolMessage(content="result", name="write_file", tool_call_id="tc1")
    assert _is_tool_message(msg)


def test_is_tool_message_for_ai_message():
    msg = AIMessage(content="hi")
    assert not _is_tool_message(msg)


# ── _emit_message_events ──────────────────────────────────────────────────────


def test_emit_message_events_tool_call():
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "write_file", "args": {"path": "auth.py", "content": "def f(): pass"}, "id": "tc1"}],
        )
        _emit_message_events(msg, "user-auth", "coder")

        events = _drain(broker)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "tool_call"
        assert e["feature"] == "user-auth"
        assert e["agent"] == "coder"
        assert e["tool"] == "write_file"
        assert "auth.py" in e["args_preview"]
    finally:
        set_progress_broker(None)


def test_emit_message_events_handoff():
    """transfer_to_<agent> tool calls become agent_handoff events, not tool_call."""
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "transfer_to_coder", "args": {}, "id": "tc1"}],
        )
        _emit_message_events(msg, "user-auth", "planner")

        events = _drain(broker)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "agent_handoff"
        assert e["from_agent"] == "planner"
        assert e["to_agent"] == "coder"
    finally:
        set_progress_broker(None)


def test_emit_message_events_tool_result():
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        msg = ToolMessage(
            content="Wrote 42 lines to auth.py",
            name="write_file",
            tool_call_id="tc1",
        )
        _emit_message_events(msg, "user-auth", "coder")

        events = _drain(broker)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "tool_result"
        assert e["tool"] == "write_file"
        assert "Wrote 42 lines" in e["result_preview"]
    finally:
        set_progress_broker(None)


def test_emit_message_events_multiple_tool_calls_in_one_message():
    """A single AIMessage with multiple tool_calls emits one event per call."""
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "query_graph", "args": {"cypher": "MATCH (n) RETURN n"}, "id": "1"},
                {"name": "write_file", "args": {"path": "x.py"}, "id": "2"},
            ],
        )
        _emit_message_events(msg, "auth", "planner")

        events = _drain(broker)
        assert len(events) == 2
        assert events[0]["tool"] == "query_graph"
        assert events[1]["tool"] == "write_file"
    finally:
        set_progress_broker(None)


def test_emit_message_events_text_only_message_emits_decision():
    """An AIMessage with text content emits an agent_decision event."""
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        msg = AIMessage(content="I'm analyzing the spec dependencies...")
        _emit_message_events(msg, "auth", "planner")

        events = _drain(broker)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "agent_decision"
        assert e["feature"] == "auth"
        assert e["agent"] == "planner"
        assert "analyzing" in e["text"]
    finally:
        set_progress_broker(None)


def test_emit_message_events_decision_then_tool_call():
    """An AIMessage with BOTH text and tool_calls emits decision + tool_call."""
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        msg = AIMessage(
            content="I need to query the graph to understand the dependencies first.",
            tool_calls=[{"name": "query_graph", "args": {"cypher": "MATCH (s:Spec) RETURN s"}, "id": "tc1"}],
        )
        _emit_message_events(msg, "auth", "planner")

        events = _drain(broker)
        types = [e["event"] for e in events]
        assert types == ["agent_decision", "tool_call"]
        assert "query the graph" in events[0]["text"]
        assert events[1]["tool"] == "query_graph"
    finally:
        set_progress_broker(None)


def test_emit_message_events_decision_with_anthropic_block_format():
    """Anthropic-style multi-block content (list of dicts) is extracted correctly."""
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        msg = AIMessage(
            content=[
                {"type": "text", "text": "Let me think about this carefully."},
                {"type": "text", "text": "I'll start by listing all specs."},
            ],
        )
        _emit_message_events(msg, "auth", "planner")

        events = _drain(broker)
        assert len(events) == 1
        assert events[0]["event"] == "agent_decision"
        assert "think about this" in events[0]["text"]
        assert "listing all specs" in events[0]["text"]
    finally:
        set_progress_broker(None)


def test_emit_message_events_decision_text_truncated():
    """Very long decision text is truncated with an ellipsis."""
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        long_text = "x" * 1000
        msg = AIMessage(content=long_text)
        _emit_message_events(msg, "auth", "planner")

        events = _drain(broker)
        assert len(events) == 1
        # 500 chars + ellipsis
        assert len(events[0]["text"]) == 501
        assert events[0]["text"].endswith("…")
    finally:
        set_progress_broker(None)


def test_emit_message_events_empty_content_no_decision():
    """An AIMessage with empty content emits no decision event (only tool calls if any)."""
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        msg = AIMessage(content="")
        _emit_message_events(msg, "auth", "planner")
        assert _drain(broker) == []
    finally:
        set_progress_broker(None)


def test_emit_message_events_whitespace_only_content_no_decision():
    """Whitespace-only content doesn't emit a decision event."""
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        msg = AIMessage(content="   \n\t  ")
        _emit_message_events(msg, "auth", "planner")
        assert _drain(broker) == []
    finally:
        set_progress_broker(None)


# ── End-to-end run_feature_swarm with mocked stream ──────────────────────────


def test_run_feature_swarm_emits_full_event_sequence():
    """A simulated LangGraph stream produces agent_active + tool_call +
    tool_result + agent_handoff events for the Logs tab."""
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        # Build a fake compiled swarm whose stream yields a realistic
        # sequence of node updates with embedded messages.
        ai_with_tool = AIMessage(
            content="I'll write the auth file",
            tool_calls=[{"name": "write_file", "args": {"path": "auth.py"}, "id": "tc1"}],
        )
        tool_result = ToolMessage(content="OK, wrote auth.py", name="write_file", tool_call_id="tc1")
        ai_handoff = AIMessage(
            content="",
            tool_calls=[{"name": "transfer_to_reviewer", "args": {}, "id": "tc2"}],
        )

        fake_chunks = [
            {"planner": {"messages": [AIMessage(content="planning auth")]}},
            {"coder": {"messages": [ai_with_tool]}},
            {"tools": {"messages": [tool_result]}},
            {"coder": {"messages": [ai_handoff]}},
            {"reviewer": {"messages": [AIMessage(content="LGTM")]}},
        ]

        mock_compiled = MagicMock()
        mock_compiled.stream.return_value = iter(fake_chunks)

        with patch("dark_factory.agents.swarm.set_current_feature"):
            result = run_feature_swarm(
                mock_compiled,
                spec_ids=["spec-1"],
                feature_name="user-auth",
            )

        assert result["status"] == "success"

        events = _drain(broker)
        event_types = [e["event"] for e in events]

        # planner active → coder active → reviewer active (3 agent transitions)
        assert event_types.count("agent_active") == 3
        # one tool_call (write_file)
        tool_calls = [e for e in events if e["event"] == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "write_file"
        assert tool_calls[0]["agent"] == "coder"
        # one tool_result (write_file done)
        tool_results = [e for e in events if e["event"] == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool"] == "write_file"
        # one handoff (coder → reviewer)
        handoffs = [e for e in events if e["event"] == "agent_handoff"]
        assert len(handoffs) == 1
        assert handoffs[0]["from_agent"] == "coder"
        assert handoffs[0]["to_agent"] == "reviewer"
    finally:
        set_progress_broker(None)


def test_run_feature_swarm_dedupes_consecutive_same_agent():
    """If the same agent appears in consecutive chunks, only one agent_active
    event is emitted (the second is deduped)."""
    broker = ProgressBroker()
    set_progress_broker(broker)
    try:
        fake_chunks = [
            {"planner": {"messages": []}},
            {"planner": {"messages": []}},  # same agent again
            {"planner": {"messages": []}},  # same agent again
            {"coder": {"messages": []}},
        ]

        mock_compiled = MagicMock()
        mock_compiled.stream.return_value = iter(fake_chunks)

        with patch("dark_factory.agents.swarm.set_current_feature"):
            run_feature_swarm(mock_compiled, spec_ids=["spec-1"], feature_name="auth")

        events = _drain(broker)
        agent_events = [e for e in events if e["event"] == "agent_active"]
        assert len(agent_events) == 2  # planner, coder
        assert agent_events[0]["agent"] == "planner"
        assert agent_events[1]["agent"] == "coder"
    finally:
        set_progress_broker(None)
