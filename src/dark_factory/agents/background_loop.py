"""Long-lived background event loop for running coroutines from sync code.

The Claude Agent SDK and other asyncio-based libraries leave background
tasks (subprocess transports, stream readers, async generators) attached
to the event loop they were created on. When that loop is then closed
(``asyncio.run`` does this on every call), pending cleanup callbacks fire
on the closed loop and raise ``RuntimeError: Event loop is closed``.

This module provides a process-wide singleton :class:`BackgroundLoop` that
keeps a single ``asyncio`` event loop running forever in a daemon thread.
Sync callers submit coroutines via :meth:`run` and block on the result.
The loop is shared across all callers, so subprocess cleanup callbacks
always have a valid loop to land on, eliminating the "Event loop is closed"
class of errors.

Usage:

.. code-block:: python

    from dark_factory.agents.background_loop import BackgroundLoop

    async def my_coro() -> str:
        ...

    result = BackgroundLoop.get().run(my_coro())
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

import structlog

log = structlog.get_logger()

T = TypeVar("T")


def _silence_closed_loop_errors(
    loop: asyncio.AbstractEventLoop,
    context: dict[str, Any],
) -> None:
    """Custom loop exception handler that swallows cleanup-stage
    ``RuntimeError('Event loop is closed')`` warnings.

    **Why this exists.** The Anthropic Python SDK's internal httpx
    ``AsyncClient`` keeps a connection pool attached to the event loop
    it was created on. When a pipeline errors and the BackgroundLoop is
    torn down, the GC eventually runs on those stale ``AsyncClient``
    instances; their ``__del__`` / ``aclose()`` tasks try to close
    their underlying ``_SelectorSocketTransport`` via
    ``loop.call_soon(self._call_connection_lost, ...)``. If the loop
    has already been closed, ``call_soon`` raises
    ``RuntimeError: Event loop is closed`` — as an
    "unretrieved task exception" the default handler dumps a full
    rich-formatted traceback to stderr.

    Those tracebacks are noisy, cosmetic, and hide the real error. This
    handler intercepts exactly that RuntimeError and drops it silently
    while forwarding every other context to the default handler.
    """
    exc = context.get("exception")
    if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
        # Silently swallow — there's nothing the operator can do and
        # no additional bookkeeping we can perform on a closed loop.
        return
    loop.default_exception_handler(context)


class BackgroundLoop:
    """Process-wide singleton: an asyncio loop running in a daemon thread."""

    _instance: "BackgroundLoop | None" = None
    _instance_lock = threading.Lock()
    # Monotonic counter of how many times the singleton has been (re)created
    # in this process. Surfaced by the metrics sampler.
    _restart_count: int = 0

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        # Install the exception handler BEFORE the thread starts so
        # any orphaned cleanup tasks that fire during early-teardown
        # also get filtered.
        self._loop.set_exception_handler(_silence_closed_loop_errors)
        self._ready = threading.Event()
        # Counter of coroutines submitted via run(). Increments are atomic
        # under the GIL for plain int attribute writes in CPython.
        self._completed_count: int = 0
        self._thread = threading.Thread(
            target=self._run_forever,
            name="dark-factory-bg-loop",
            daemon=True,
        )
        self._thread.start()
        # Wait for the loop to actually start spinning before returning
        self._ready.wait(timeout=5.0)
        if not self._ready.is_set():
            raise RuntimeError("BackgroundLoop failed to start within 5s")
        log.debug("background_loop_started")

    def _run_forever(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            try:
                self._loop.close()
            except Exception:  # pragma: no cover
                pass

    @classmethod
    def get(cls) -> "BackgroundLoop":
        """Return the process-wide singleton, creating it on first call."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
                cls._restart_count += 1
            return cls._instance

    @classmethod
    def restart_count(cls) -> int:
        """Return how many times the singleton has been (re)started."""
        return cls._restart_count

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton. Drains any pending tasks before stopping
        the loop, so subprocess cleanup callbacks have a chance to complete.
        The next ``get()`` call creates a fresh instance.
        """
        with cls._instance_lock:
            if cls._instance is None:
                return
            instance = cls._instance
            cls._instance = None

        # Try to gracefully cancel + drain outstanding tasks before stopping.
        # Two-phase drain: (1) cancel and await currently-pending tasks,
        # (2) yield to the loop a few times so any ``call_soon``
        # callbacks scheduled by __del__ / aclose() during the cancel
        # phase also get a chance to run. Without phase 2, httpx
        # cleanup callbacks would fire on a closed loop and produce
        # "Event loop is closed" traceback spam.
        if instance._loop.is_running():
            async def _drain() -> None:
                tasks = [
                    t for t in asyncio.all_tasks(instance._loop)
                    if not t.done() and t is not asyncio.current_task()
                ]
                for t in tasks:
                    t.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                # Yield control several times so callbacks scheduled by
                # transport cleanup (which itself may schedule more
                # callbacks) all fire before we stop the loop.
                for _ in range(5):
                    await asyncio.sleep(0)

            try:
                future = asyncio.run_coroutine_threadsafe(_drain(), instance._loop)
                future.result(timeout=5.0)
            except Exception as exc:  # pragma: no cover — best-effort cleanup
                log.warning("background_loop_drain_failed", error=str(exc))

            try:
                instance._loop.call_soon_threadsafe(instance._loop.stop)
            except RuntimeError:
                pass

        instance._thread.join(timeout=5.0)
        log.debug("background_loop_stopped")

    def run(self, coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
        """Run a coroutine on the background loop and block until it completes.

        Safe to call from any thread (including threads that already have
        their own event loop). The coroutine executes on the shared loop, so
        any background tasks/subprocess cleanup it spawns can land on a
        loop that stays alive for the lifetime of the process.

        :param coro: the coroutine to run
        :param timeout: optional max wait in seconds (None = wait forever)
        :raises Exception: re-raises any exception the coroutine raised
        """
        if not self._loop.is_running():
            raise RuntimeError("BackgroundLoop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        finally:
            self._completed_count += 1

    def task_counts(self) -> tuple[int, int, int]:
        """Return ``(active, pending, completed)`` task counts on this loop.

        ``active`` = tasks currently executing (not done).
        ``pending`` = tasks scheduled but not started (approximated as 0 since
        asyncio doesn't expose a separate "pending" bucket — we conflate with
        active for simplicity).
        ``completed`` = cumulative total of coroutines that have finished via
        :meth:`run`.
        """
        if not self._loop.is_running():
            return (0, 0, self._completed_count)
        try:
            all_tasks = asyncio.all_tasks(self._loop)
        except Exception:  # pragma: no cover — best-effort
            return (0, 0, self._completed_count)
        active = sum(1 for t in all_tasks if not t.done())
        return (active, 0, self._completed_count)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Direct access to the underlying loop (for advanced use cases)."""
        return self._loop

    @property
    def is_running(self) -> bool:
        return self._loop.is_running()
