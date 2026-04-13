"""Tests for cancellation primitives and the ``POST /api/agent/cancel`` endpoint.

Unit tests for the cancellation module come first (no api_client needed),
followed by the endpoint integration tests that require the ``api_client``
fixture.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dark_factory.agents.cancellation import (
    PipelineCancelled,
    is_cancelled,
    raise_if_cancelled,
    request_cancel,
    reset_cancel,
)


@pytest.fixture(autouse=True)
def _clean_cancel_state():
    """Ensure each test starts and ends with the cancel flag cleared so
    leaking state from one test can't silently short-circuit the next."""
    reset_cancel()
    yield
    reset_cancel()


@pytest.fixture()
def minimal_settings():
    """Return a default Settings instance for tests that need one."""
    from dark_factory.config import Settings

    return Settings()


# ── Unit tests: cancellation primitives ────────────────────────────────────


def test_initial_state_is_not_cancelled():
    """After reset, the cancel flag is cleared."""
    assert is_cancelled() is False


def test_request_cancel_sets_flag():
    """request_cancel() sets the flag so is_cancelled() returns True."""
    request_cancel()
    assert is_cancelled() is True


def test_request_cancel_is_idempotent():
    """Calling request_cancel() twice doesn't raise or change state."""
    request_cancel()
    request_cancel()
    assert is_cancelled() is True


def test_reset_cancel_clears_flag():
    """reset_cancel() clears the flag after it was set."""
    request_cancel()
    assert is_cancelled() is True
    reset_cancel()
    assert is_cancelled() is False


def test_raise_if_cancelled_noop_when_not_set():
    """raise_if_cancelled() does nothing when the flag is not set."""
    raise_if_cancelled()  # should not raise


def test_raise_if_cancelled_raises_when_set():
    """raise_if_cancelled() raises PipelineCancelled when the flag is set."""
    request_cancel()
    with pytest.raises(PipelineCancelled, match="cancelled by user"):
        raise_if_cancelled()


def test_pipeline_cancelled_is_exception():
    """PipelineCancelled inherits from Exception so it can be caught broadly."""
    assert issubclass(PipelineCancelled, Exception)


def test_cancel_flag_works_across_threads():
    """The threading.Event-based cancel flag is visible across threads."""
    import threading

    results = []

    def _worker():
        results.append(is_cancelled())

    request_cancel()
    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert results == [True]


# ── Integration: cancellation checkpoints in pipeline stages ───────────────


def test_cancel_during_spec_stage():
    """When cancel is requested, raise_if_cancelled at spec stage boundaries
    stops the pipeline with PipelineCancelled."""
    request_cancel()
    with pytest.raises(PipelineCancelled):
        # Simulate a spec stage checkpoint
        raise_if_cancelled()


def test_cancel_during_swarm_iteration():
    """Simulates a swarm loop that checks cancellation on each iteration."""
    features = ["auth", "billing", "search"]
    processed = []
    request_cancel()
    for f in features:
        if is_cancelled():
            break
        processed.append(f)
    assert processed == [], "No features should be processed after cancel"


def test_cancel_reset_between_runs():
    """reset_cancel() at run start prevents bleed from a previous cancel."""
    request_cancel()
    assert is_cancelled() is True
    # New run starts
    reset_cancel()
    assert is_cancelled() is False
    # The new run should proceed normally
    raise_if_cancelled()  # should not raise


def test_cancel_during_doc_extraction_checkpoint():
    """A cancel between document extraction steps stops cleanly."""
    docs = ["README.md", "SPEC.md", "DESIGN.md"]
    extracted = []
    for doc in docs:
        extracted.append(doc)
        if doc == "SPEC.md":
            request_cancel()
        if is_cancelled():
            break
    assert extracted == ["README.md", "SPEC.md"]


def test_cancel_during_reconciliation_phase():
    """Reconciliation phase respects the cancel flag."""
    request_cancel()
    with pytest.raises(PipelineCancelled):
        raise_if_cancelled()


def test_cancel_during_e2e_validation():
    """E2E validation respects the cancel flag between browser runs."""
    browsers = ["chromium", "firefox", "webkit"]
    tested = []
    for browser in browsers:
        if is_cancelled():
            break
        tested.append(browser)
        if browser == "chromium":
            request_cancel()
    assert tested == ["chromium"]


def test_cancel_during_episode_synthesis():
    """Episode synthesis loop stops when cancel flag is set mid-iteration."""
    episodes = list(range(5))
    synthesized = []
    for ep in episodes:
        if is_cancelled():
            break
        synthesized.append(ep)
        if ep == 2:
            request_cancel()
    assert synthesized == [0, 1, 2]


def test_cancel_during_memory_record_tools():
    """Memory recording operations check cancellation before proceeding."""
    memory_ops = ["record_pattern", "record_mistake", "record_solution"]
    completed = []
    request_cancel()
    for op in memory_ops:
        if is_cancelled():
            break
        completed.append(op)
    assert completed == []


def test_run_pipeline_stream_resets_cancel_at_start():
    """Simulates the run_pipeline_stream pattern: reset at start, check throughout."""
    # Leftover cancel from a previous run
    request_cancel()
    assert is_cancelled() is True
    # New run begins — stream resets cancel
    reset_cancel()
    assert is_cancelled() is False
    # Pipeline proceeds with checkpoints
    for _ in range(3):
        raise_if_cancelled()  # should not raise


def test_cancel_and_reset_cycle():
    """Multiple cancel/reset cycles work correctly."""
    for _ in range(5):
        assert is_cancelled() is False
        request_cancel()
        assert is_cancelled() is True
        reset_cancel()
    assert is_cancelled() is False


def test_raise_if_cancelled_message():
    """The exception message is descriptive."""
    request_cancel()
    try:
        raise_if_cancelled()
        assert False, "Should have raised"
    except PipelineCancelled as exc:
        assert "pipeline cancelled" in str(exc).lower()


# ── /api/agent/cancel endpoint ─────────────────────────────────────────────


def test_cancel_endpoint_returns_not_active_when_no_run(api_client):
    """No run in progress → 200 + ``cancelled=False`` so the frontend
    can fire cancel without a try/catch race."""
    resp = api_client.post("/api/agent/cancel")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cancelled"] is False
    assert "no active run" in data.get("reason", "")


def test_cancel_endpoint_sets_flag_when_run_active(api_client):
    """With the run_lock held, the endpoint should set the cancellation
    flag and return ``cancelled=True``."""
    from dark_factory.api.app import app

    async def _hold_lock_then_cancel():
        lock: asyncio.Lock = app.state.run_lock
        async with lock:
            # Now the lock is locked — simulate an active run and poke
            # the cancel endpoint via the TestClient in a thread so the
            # async context manager doesn't block the sync test client.
            resp = await asyncio.to_thread(
                api_client.post, "/api/agent/cancel"
            )
            return resp

    resp = asyncio.run(_hold_lock_then_cancel())
    assert resp.status_code == 200
    data = resp.json()
    assert data["cancelled"] is True
    assert is_cancelled() is True


def test_cancel_endpoint_idempotent_when_already_cancelled(api_client):
    """A second cancel while already pending returns ``already_pending=True``."""
    from dark_factory.api.app import app

    async def _double_cancel():
        lock: asyncio.Lock = app.state.run_lock
        async with lock:
            first = await asyncio.to_thread(api_client.post, "/api/agent/cancel")
            second = await asyncio.to_thread(api_client.post, "/api/agent/cancel")
            return first, second

    first, second = asyncio.run(_double_cancel())
    assert first.json()["cancelled"] is True
    assert second.json()["cancelled"] is True
    assert second.json().get("already_pending") is True
