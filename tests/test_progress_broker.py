"""Tests for ProgressBroker and the /api/agent/events SSE endpoint.

Unit tests for the broker come first (no api_client needed), followed
by the endpoint integration tests that require the ``api_client`` fixture.
"""

from __future__ import annotations

import asyncio

import pytest

from dark_factory.agents.progress import ProgressBroker


# ── Unit tests: ProgressBroker ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_returns_queue():
    """subscribe() returns an asyncio.Queue bound to the running loop."""
    broker = ProgressBroker()
    queue = broker.subscribe()
    assert isinstance(queue, asyncio.Queue)
    assert broker.subscriber_count == 1


@pytest.mark.asyncio
async def test_publish_fans_out_to_all_subscribers():
    """publish() delivers the event to every subscriber's queue."""
    broker = ProgressBroker()
    q1 = broker.subscribe(include_history=False)
    q2 = broker.subscribe(include_history=False)
    broker.publish({"event": "step_complete", "step": 1})
    # Give the event loop a chance to process call_soon_threadsafe
    await asyncio.sleep(0.05)
    msg1 = q1.get_nowait()
    msg2 = q2.get_nowait()
    assert msg1["event"] == "step_complete"
    assert msg2["event"] == "step_complete"


@pytest.mark.asyncio
async def test_history_replay_on_subscribe():
    """New subscribers receive history events if include_history=True."""
    broker = ProgressBroker()
    broker.publish({"event": "old_event_1"})
    broker.publish({"event": "old_event_2"})
    queue = broker.subscribe(include_history=True)
    assert queue.qsize() == 2
    first = queue.get_nowait()
    assert first["event"] == "old_event_1"


@pytest.mark.asyncio
async def test_subscribe_without_history():
    """include_history=False gives an empty queue."""
    broker = ProgressBroker()
    broker.publish({"event": "old_event"})
    queue = broker.subscribe(include_history=False)
    assert queue.empty()


@pytest.mark.asyncio
async def test_unsubscribe_removes_queue():
    """After unsubscribe, the queue no longer receives events."""
    broker = ProgressBroker()
    queue = broker.subscribe(include_history=False)
    assert broker.subscriber_count == 1
    broker.unsubscribe(queue)
    assert broker.subscriber_count == 0


@pytest.mark.asyncio
async def test_clear_history():
    """clear_history() empties the history ring buffer."""
    broker = ProgressBroker()
    broker.publish({"event": "e1"})
    broker.publish({"event": "e2"})
    assert broker.history_count == 2
    broker.clear_history()
    assert broker.history_count == 0


@pytest.mark.asyncio
async def test_queue_full_drops_oldest():
    """When a subscriber's queue is full, the oldest event is dropped."""
    broker = ProgressBroker(queue_size=2)
    queue = broker.subscribe(include_history=False)
    broker.publish({"event": "e1"})
    broker.publish({"event": "e2"})
    await asyncio.sleep(0.05)
    # Queue is now full (2 items). Next publish should drop oldest.
    broker.publish({"event": "e3"})
    await asyncio.sleep(0.05)
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    events = [i["event"] for i in items]
    # e1 should have been dropped, e2 and e3 remain
    assert "e3" in events
    assert len(items) == 2


@pytest.mark.asyncio
async def test_publish_adds_timestamp():
    """publish() adds a timestamp field to the event."""
    broker = ProgressBroker()
    queue = broker.subscribe(include_history=False)
    broker.publish({"event": "test"})
    await asyncio.sleep(0.05)
    msg = queue.get_nowait()
    assert "timestamp" in msg
    assert isinstance(msg["timestamp"], float)


@pytest.mark.asyncio
async def test_history_size_limit():
    """History ring buffer respects the configured max size."""
    broker = ProgressBroker(history_size=3)
    for i in range(5):
        broker.publish({"event": f"e{i}"})
    assert broker.history_count == 3


@pytest.mark.asyncio
async def test_emit_progress_integration():
    """emit_progress() publishes through the global broker."""
    from dark_factory.agents import tools as tools_mod

    broker = ProgressBroker()
    original = tools_mod._progress_broker
    tools_mod._progress_broker = broker
    try:
        queue = broker.subscribe(include_history=False)
        tools_mod.emit_progress("test_event", detail="hello")
        await asyncio.sleep(0.05)
        msg = queue.get_nowait()
        assert msg["event"] == "test_event"
        assert msg["detail"] == "hello"
    finally:
        tools_mod._progress_broker = original


def test_subscribe_outside_event_loop_raises():
    """subscribe() raises RuntimeError if not called from an event loop."""
    broker = ProgressBroker()
    with pytest.raises(RuntimeError, match="running event loop"):
        broker.subscribe()


# ── SSE endpoint registration ────────────────────────────────────────────────


def test_agent_events_in_openapi_schema(api_client):
    """The new endpoint is registered in the OpenAPI schema."""
    resp = api_client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/api/agent/events" in paths


def test_app_state_has_progress_broker(api_client):
    """The app lifespan installs a broker on app.state."""
    from dark_factory.api.app import app

    assert app.state.progress_broker is not None
    assert isinstance(app.state.progress_broker, ProgressBroker)


def test_app_state_has_run_lock(api_client):
    """C1: the lifespan creates an asyncio.Lock to serialize runs."""
    import asyncio as _asyncio

    from dark_factory.api.app import app

    assert isinstance(app.state.run_lock, _asyncio.Lock)


def test_concurrent_run_returns_409(api_client):
    """C1: a second agent/run request is rejected with 409 while the first is running."""
    from dark_factory.api.app import app

    # Simulate an in-progress run by acquiring the lock directly
    app.state.run_lock._locked = True  # type: ignore[attr-defined]
    try:
        resp = api_client.post(
            "/api/agent/run",
            json={"requirements_path": "./openspec"},
        )
        assert resp.status_code == 409
        assert "already in progress" in resp.json()["detail"].lower()
    finally:
        app.state.run_lock._locked = False  # type: ignore[attr-defined]
