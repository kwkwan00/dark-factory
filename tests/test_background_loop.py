"""Tests for the BackgroundLoop singleton used by deep agents."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from dark_factory.agents.background_loop import BackgroundLoop


@pytest.fixture(autouse=True)
def _reset_background_loop():
    """Make sure each test starts with a fresh singleton (no leakage)."""
    BackgroundLoop.reset()
    yield
    BackgroundLoop.reset()


def test_get_returns_singleton():
    """Multiple get() calls return the same instance."""
    a = BackgroundLoop.get()
    b = BackgroundLoop.get()
    assert a is b


def test_loop_is_running_after_get():
    bg = BackgroundLoop.get()
    assert bg.is_running


def test_run_simple_coroutine_returns_value():
    bg = BackgroundLoop.get()

    async def _coro():
        return 42

    result = bg.run(_coro())
    assert result == 42


def test_run_propagates_exceptions():
    bg = BackgroundLoop.get()

    async def _bad():
        raise ValueError("explicit failure")

    with pytest.raises(ValueError, match="explicit failure"):
        bg.run(_bad())


def test_run_handles_async_generator_pattern():
    """The Claude SDK uses async generators internally — verify they work."""
    bg = BackgroundLoop.get()

    async def _agen():
        for i in range(3):
            yield i

    async def _consume():
        items = []
        async for v in _agen():
            items.append(v)
        return items

    result = bg.run(_consume())
    assert result == [0, 1, 2]


def test_run_from_multiple_threads_concurrently():
    """The loop is shared across threads — submissions interleave correctly."""
    bg = BackgroundLoop.get()
    results: list[int] = []
    lock = threading.Lock()

    def _worker(n: int):
        async def _coro():
            await asyncio.sleep(0.01)
            return n * n

        value = bg.run(_coro())
        with lock:
            results.append(value)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [i * i for i in range(10)]


def test_run_preserves_loop_across_calls():
    """The loop is the SAME loop across multiple .run() calls (no closure)."""
    bg = BackgroundLoop.get()

    captured_loops: list[asyncio.AbstractEventLoop] = []

    async def _capture():
        captured_loops.append(asyncio.get_running_loop())
        return None

    bg.run(_capture())
    bg.run(_capture())
    bg.run(_capture())

    assert len(captured_loops) == 3
    # All three runs landed on the same loop
    assert captured_loops[0] is captured_loops[1]
    assert captured_loops[1] is captured_loops[2]


def test_subprocess_cleanup_callbacks_dont_raise():
    """Simulate the Claude SDK pattern: an async generator that yields
    items and schedules a cleanup callback. The callback must not see
    'Event loop is closed'."""
    bg = BackgroundLoop.get()
    cleanup_calls: list[bool] = []

    async def _streaming_op():
        # Schedule a cleanup callback that runs after this coroutine ends
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, lambda: cleanup_calls.append(True))
        # Do some work
        await asyncio.sleep(0.01)
        return "ok"

    result = bg.run(_streaming_op())
    assert result == "ok"
    # Wait for the deferred callback to fire — it must NOT raise
    time.sleep(0.1)
    assert cleanup_calls == [True]


def test_reset_creates_fresh_instance():
    """reset() tears down the singleton; the next get() builds a new one."""
    a = BackgroundLoop.get()
    BackgroundLoop.reset()
    b = BackgroundLoop.get()
    assert a is not b
    assert b.is_running


def test_run_without_loop_running_raises():
    """If somehow the singleton's loop has stopped, .run() raises clearly."""
    bg = BackgroundLoop.get()
    bg._loop.call_soon_threadsafe(bg._loop.stop)
    bg._thread.join(timeout=2.0)

    async def _coro():
        return 1

    coro = _coro()
    try:
        with pytest.raises(RuntimeError, match="not running"):
            bg.run(coro)
    finally:
        coro.close()  # close the coroutine so we don't get a RuntimeWarning


# ── Custom loop exception handler ─────────────────────────────────────────


def test_silence_closed_loop_errors_swallows_runtime_error() -> None:
    """The custom exception handler must swallow the specific
    ``RuntimeError('Event loop is closed')`` that orphaned httpx /
    Anthropic SDK cleanup tasks emit after the BackgroundLoop has been
    torn down. Regression guard: without this, the error dumps a giant
    rich-formatted traceback to stderr on every failed pipeline run,
    hiding the real error above it."""
    from dark_factory.agents.background_loop import _silence_closed_loop_errors

    mock_loop = _RecordingLoop()
    exc = RuntimeError("Event loop is closed")
    context = {"message": "Task exception was never retrieved", "exception": exc}

    _silence_closed_loop_errors(mock_loop, context)

    # The default handler MUST NOT have been called — the error was
    # silently swallowed.
    assert mock_loop.default_handler_calls == []


def test_silence_closed_loop_errors_forwards_other_exceptions() -> None:
    """Only the specific "Event loop is closed" RuntimeError is
    swallowed. Every other exception type (including other
    RuntimeErrors with different messages) must still reach the
    default handler so real bugs aren't hidden."""
    from dark_factory.agents.background_loop import _silence_closed_loop_errors

    cases = [
        RuntimeError("some unrelated runtime error"),
        ValueError("bad value"),
        KeyError("missing"),
        Exception("plain exception"),
    ]
    for exc in cases:
        mock_loop = _RecordingLoop()
        _silence_closed_loop_errors(
            mock_loop, {"message": "oops", "exception": exc}
        )
        assert len(mock_loop.default_handler_calls) == 1, (
            f"expected forward for {type(exc).__name__}, got nothing"
        )


def test_silence_closed_loop_errors_forwards_when_no_exception_key() -> None:
    """Contexts without an exception key (rare but possible — e.g. a
    "Task was destroyed but it is pending!" warning) must still reach
    the default handler."""
    from dark_factory.agents.background_loop import _silence_closed_loop_errors

    mock_loop = _RecordingLoop()
    _silence_closed_loop_errors(
        mock_loop, {"message": "Task was destroyed but it is pending"}
    )
    assert len(mock_loop.default_handler_calls) == 1


def test_background_loop_installs_exception_handler_at_init() -> None:
    """End-to-end: a fresh ``BackgroundLoop.get()`` has the custom
    handler installed on its underlying loop. This is the integration
    test that ties the helper function to the class."""
    from dark_factory.agents.background_loop import _silence_closed_loop_errors

    bg = BackgroundLoop.get()
    assert bg.loop.get_exception_handler() is _silence_closed_loop_errors


class _RecordingLoop:
    """Minimal fake loop that records calls to ``default_exception_handler``.

    Just enough surface for ``_silence_closed_loop_errors`` to decide
    whether to forward a context.
    """

    def __init__(self) -> None:
        self.default_handler_calls: list[dict] = []

    def default_exception_handler(self, context: dict) -> None:
        self.default_handler_calls.append(context)
