"""Tests for emit_progress + the AG-UI bridge _translate_progress helper."""

from __future__ import annotations

import asyncio
import json

from ag_ui.encoder import EventEncoder

from dark_factory.agents.progress import ProgressBroker
from dark_factory.agents.tools import emit_progress, set_progress_broker
from dark_factory.api.ag_ui_bridge import _translate_progress


# ── emit_progress ─────────────────────────────────────────────────────────────


def test_emit_progress_is_noop_when_no_broker():
    """emit_progress is safe to call when no broker is installed."""
    set_progress_broker(None)
    emit_progress("feature_started", feature="auth")  # should not raise


def test_emit_progress_forwards_to_broker():
    """emit_progress publishes through the installed broker."""
    async def _run():
        broker = ProgressBroker()
        set_progress_broker(broker)
        try:
            queue = broker.subscribe(include_history=False)
            emit_progress("feature_started", feature="auth", spec_count=3)
            await asyncio.sleep(0.01)
            event = queue.get_nowait()
            # L6 fix: subset assertion — new auto-injected fields won't break this
            assert event["event"] == "feature_started"
            assert event["feature"] == "auth"
            assert event["spec_count"] == 3
            assert "timestamp" in event
        finally:
            set_progress_broker(None)

    asyncio.run(_run())


def test_emit_progress_swallows_broker_exceptions():
    """A broken broker must never propagate exceptions to callers."""
    class BrokenBroker:
        def publish(self, event):
            raise RuntimeError("boom")

    set_progress_broker(BrokenBroker())
    try:
        emit_progress("feature_started")  # must not raise
    finally:
        set_progress_broker(None)


# ── _translate_progress helper ────────────────────────────────────────────────


def _collect(gen) -> list[str]:
    """_translate_progress is now a synchronous generator (H3 fix)."""
    return list(gen)


def _parse_event(sse: str) -> dict:
    for line in sse.split("\n"):
        if line.startswith("data: "):
            return json.loads(line[6:])
    return {}


def test_translate_layer_started_emits_text_message():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "layer_started",
                "layer": 1,
                "total_layers": 2,
                "features": ["auth", "db"],
            },
            {},
            {},
        )
    )
    assert len(chunks) == 3
    content = _parse_event(chunks[1])
    assert content["type"] == "TEXT_MESSAGE_CONTENT"
    assert "Layer 1/2" in content["delta"]
    assert "auth" in content["delta"]
    assert "db" in content["delta"]


def test_translate_feature_started_emits_step_started_and_text():
    encoder = EventEncoder()
    feature_step_ids: dict[str, str] = {}
    chunks = _collect(
        _translate_progress(
            encoder,
            {"event": "feature_started", "feature": "auth", "spec_count": 3},
            feature_step_ids,
            {},
        )
    )
    types = [_parse_event(c).get("type") for c in chunks]
    assert "STEP_STARTED" in types
    assert "TEXT_MESSAGE_CONTENT" in types
    assert "auth" in feature_step_ids


def test_translate_feature_completed_emits_step_finished():
    encoder = EventEncoder()
    feature_step_ids = {"auth": "step-abc-123"}
    last_agent = {"auth": "coder"}
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "feature_completed",
                "feature": "auth",
                "status": "success",
                "artifacts": 5,
                "tests": 3,
            },
            feature_step_ids,
            last_agent,
        )
    )
    types = [_parse_event(c).get("type") for c in chunks]
    assert "STEP_FINISHED" in types
    assert "auth" not in feature_step_ids
    assert "auth" not in last_agent


def test_translate_feature_completed_includes_error():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "feature_completed",
                "feature": "auth",
                "status": "error",
                "artifacts": 0,
                "tests": 0,
                "error": "LLM timeout",
            },
            {},
            {},
        )
    )
    content_chunks = [_parse_event(c) for c in chunks]
    texts = [c.get("delta", "") for c in content_chunks if c.get("type") == "TEXT_MESSAGE_CONTENT"]
    combined = " ".join(texts)
    assert "LLM timeout" in combined
    assert "error" in combined


def test_translate_agent_active_dedupes_same_agent():
    """When the same agent fires twice in a row, only emit one text message."""
    encoder = EventEncoder()
    last_agent: dict[str, str] = {}

    chunks1 = _collect(
        _translate_progress(
            encoder,
            {"event": "agent_active", "feature": "auth", "agent": "planner", "messages": 1},
            {},
            last_agent,
        )
    )
    chunks2 = _collect(
        _translate_progress(
            encoder,
            {"event": "agent_active", "feature": "auth", "agent": "planner", "messages": 2},
            {},
            last_agent,
        )
    )
    assert len(chunks1) == 3
    assert len(chunks2) == 0


def test_translate_agent_active_emits_on_change():
    """When the agent changes, a new text message is emitted."""
    encoder = EventEncoder()
    last_agent: dict[str, str] = {}

    _collect(
        _translate_progress(
            encoder,
            {"event": "agent_active", "feature": "auth", "agent": "planner", "messages": 1},
            {},
            last_agent,
        )
    )
    chunks = _collect(
        _translate_progress(
            encoder,
            {"event": "agent_active", "feature": "auth", "agent": "coder", "messages": 2},
            {},
            last_agent,
        )
    )
    assert len(chunks) == 3
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    assert "coder" in combined


def test_translate_feature_skipped():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {"event": "feature_skipped", "feature": "api", "reason": "dependency 'auth' failed"},
            {},
            {},
        )
    )
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    assert "api" in combined
    assert "dependency" in combined


def test_translate_agent_decision():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "agent_decision",
                "feature": "user-auth",
                "agent": "planner",
                "text": "I will analyze the dependencies first by querying the graph.",
            },
            {},
            {},
        )
    )
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    assert "user-auth" in combined
    assert "planner" in combined
    assert "analyze the dependencies" in combined


def test_translate_agent_decision_truncates_long_text():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "agent_decision",
                "feature": "f",
                "agent": "coder",
                "text": "x" * 500,
            },
            {},
            {},
        )
    )
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    # Should be truncated to 200 chars + ellipsis
    assert "…" in combined


def test_translate_agent_handoff():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "agent_handoff",
                "feature": "user-auth",
                "from_agent": "planner",
                "to_agent": "coder",
            },
            {},
            {},
        )
    )
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    assert "user-auth" in combined
    assert "planner" in combined
    assert "coder" in combined
    assert "→" in combined or "->" in combined


def test_translate_tool_call_with_args():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "tool_call",
                "feature": "user-auth",
                "agent": "coder",
                "tool": "write_file",
                "args_preview": "{'path': 'auth.py'}",
            },
            {},
            {},
        )
    )
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    assert "coder" in combined
    assert "write_file" in combined
    assert "auth.py" in combined


def test_translate_tool_call_no_args():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "tool_call",
                "feature": "f",
                "agent": "tester",
                "tool": "list_specs",
            },
            {},
            {},
        )
    )
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    assert "list_specs" in combined
    assert "tester" in combined


def test_translate_tool_result():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "tool_result",
                "feature": "f",
                "tool": "write_file",
                "result_preview": "Wrote 42 lines",
            },
            {},
            {},
        )
    )
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    assert "write_file" in combined
    assert "Wrote 42 lines" in combined


def test_translate_tool_result_truncates_long_preview():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "tool_result",
                "feature": "f",
                "tool": "query_graph",
                "result_preview": "x" * 200,
            },
            {},
            {},
        )
    )
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    # 80 chars + ellipsis
    assert "…" in combined


def test_translate_eval_rubric_renders_each_metric():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "eval_rubric",
                "feature": "spec-generation",
                "requirement_id": "req-1",
                "requirement_title": "User Auth",
                "attempt": 2,
                "max_handoffs": 5,
                "avg_score": 0.78,
                "threshold": 0.8,
                "metrics": [
                    {"name": "Spec Correctness", "score": 0.85, "passed": True, "reason": "logic ok"},
                    {"name": "Spec Coherence", "score": 0.71, "passed": False, "reason": "vague"},
                ],
            },
            {},
            {},
        )
    )
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    # Header has feature/title/attempt/avg/threshold
    assert "User Auth" in combined
    assert "2/5" in combined
    assert "0.78" in combined
    # Each metric line is rendered with score + status marker
    assert "Spec Correctness" in combined
    assert "Spec Coherence" in combined
    assert "0.85" in combined
    assert "0.71" in combined
    # Reasons appear in some form
    assert "logic ok" in combined
    assert "vague" in combined


def test_translate_eval_rubric_truncates_long_reason():
    encoder = EventEncoder()
    long_reason = "x" * 300
    chunks = _collect(
        _translate_progress(
            encoder,
            {
                "event": "eval_rubric",
                "requirement_title": "F",
                "attempt": 1,
                "max_handoffs": 1,
                "avg_score": 0.5,
                "threshold": 0.8,
                "metrics": [
                    {"name": "M", "score": 0.5, "passed": False, "reason": long_reason},
                ],
            },
            {},
            {},
        )
    )
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    # Should be truncated
    assert "…" in combined


def test_translate_unknown_event_is_noop():
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {"event": "mystery_event", "feature": "x"},
            {},
            {},
        )
    )
    assert chunks == []


def test_translate_missing_fields_uses_placeholder():
    """L2 fix: missing fields render as '(unknown)' not '?'."""
    encoder = EventEncoder()
    chunks = _collect(
        _translate_progress(
            encoder,
            {"event": "feature_started"},  # no feature name
            {},
            {},
        )
    )
    combined = " ".join(_parse_event(c).get("delta", "") for c in chunks)
    assert "(unknown)" in combined
