"""Tests for the ProgressBroker fan-out and /api/agent/events SSE endpoint."""

from __future__ import annotations

import asyncio
import threading

from dark_factory.agents.progress import ProgressBroker


# ── Broker unit tests ────────────────────────────────────────────────────────


def test_broker_subscribe_returns_queue():
    async def _run():
        broker = ProgressBroker()
        q = broker.subscribe()
        assert isinstance(q, asyncio.Queue)
        assert broker.subscriber_count == 1
        broker.unsubscribe(q)
        assert broker.subscriber_count == 0

    asyncio.run(_run())


def test_broker_subscribe_outside_loop_raises():
    """C3: subscribe() must be called from a running event loop."""
    import pytest as _pytest

    broker = ProgressBroker()
    with _pytest.raises(RuntimeError, match="running event loop"):
        broker.subscribe()


def test_broker_publish_fans_out_to_all_subscribers():
    async def _run():
        broker = ProgressBroker()
        q1 = broker.subscribe(include_history=False)
        q2 = broker.subscribe(include_history=False)

        broker.publish({"event": "layer_started", "layer": 1})
        await asyncio.sleep(0.01)

        assert q1.qsize() == 1
        assert q2.qsize() == 1
        e1 = q1.get_nowait()
        assert e1["event"] == "layer_started"
        assert "timestamp" in e1

    asyncio.run(_run())


def test_broker_history_replay_on_new_subscriber():
    async def _run():
        broker = ProgressBroker(history_size=10)
        broker.publish({"event": "feature_started", "feature": "auth"})
        broker.publish({"event": "feature_completed", "feature": "auth"})
        await asyncio.sleep(0.01)

        q = broker.subscribe(include_history=True)
        events = []
        while not q.empty():
            events.append(q.get_nowait())

        assert len(events) == 2
        assert events[0]["event"] == "feature_started"
        assert events[1]["event"] == "feature_completed"

    asyncio.run(_run())


def test_broker_subscribe_without_history():
    async def _run():
        broker = ProgressBroker()
        broker.publish({"event": "a"})
        broker.publish({"event": "b"})
        await asyncio.sleep(0.01)

        q = broker.subscribe(include_history=False)
        assert q.empty()

    asyncio.run(_run())


def test_broker_publish_from_thread_is_delivered():
    """Events published from a worker thread reach async subscribers."""

    async def _run():
        broker = ProgressBroker()
        q = broker.subscribe(include_history=False)

        def _worker():
            broker.publish({"event": "from_thread"})

        t = threading.Thread(target=_worker)
        t.start()
        t.join()

        event = await asyncio.wait_for(q.get(), timeout=1.0)
        assert event["event"] == "from_thread"

    asyncio.run(_run())


def test_broker_history_bounded_by_size():
    async def _run():
        broker = ProgressBroker(history_size=3)
        for i in range(10):
            broker.publish({"event": "e", "n": i})
        await asyncio.sleep(0.01)

        assert broker.history_count == 3
        q = broker.subscribe(include_history=True)
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        assert [e["n"] for e in events] == [7, 8, 9]

    asyncio.run(_run())


def test_broker_unsubscribe_stops_delivery():
    async def _run():
        broker = ProgressBroker()
        q = broker.subscribe(include_history=False)
        broker.unsubscribe(q)

        broker.publish({"event": "not_delivered"})
        await asyncio.sleep(0.01)

        assert q.empty()

    asyncio.run(_run())


def test_broker_clear_history():
    """L1: clear_history() drops all historical events."""
    async def _run():
        broker = ProgressBroker()
        broker.publish({"event": "a"})
        broker.publish({"event": "b"})
        await asyncio.sleep(0.01)
        assert broker.history_count == 2

        broker.clear_history()
        assert broker.history_count == 0

        q = broker.subscribe(include_history=True)
        assert q.empty()

    asyncio.run(_run())


def test_broker_queue_full_drops_oldest():
    """H2-related: when a subscriber queue is full, the oldest event is dropped."""
    async def _run():
        broker = ProgressBroker(queue_size=3)
        q = broker.subscribe(include_history=False)

        for i in range(10):
            broker.publish({"event": "e", "n": i})
        await asyncio.sleep(0.01)

        # Queue should hold at most 3 items — the most recent ones
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        assert len(events) <= 3
        # The last event should still be present
        assert any(e["n"] == 9 for e in events)

    asyncio.run(_run())


# ── emit_progress integration with broker ────────────────────────────────────


def test_emit_progress_publishes_to_broker():
    """emit_progress forwards to an installed broker."""
    from dark_factory.agents.tools import emit_progress, set_progress_broker

    async def _run():
        broker = ProgressBroker()
        set_progress_broker(broker)
        try:
            q = broker.subscribe(include_history=False)
            emit_progress("feature_started", feature="auth")
            await asyncio.sleep(0.01)

            event = q.get_nowait()
            assert event["event"] == "feature_started"
            assert event["feature"] == "auth"
        finally:
            set_progress_broker(None)

    asyncio.run(_run())


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
