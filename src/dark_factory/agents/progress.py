"""Thread-safe progress event broker.

Swarm worker threads call :func:`dark_factory.agents.tools.emit_progress`
which publishes to the broker. The broker fans out events to all active
subscribers. Each subscriber owns an :class:`asyncio.Queue` bound to the
event loop that created it (so multi-loop deployments are handled correctly).

Typical subscribers:
- ``ag_ui_bridge.run_pipeline_stream`` — consumes events for the currently
  running pipeline so the Run Pipeline tab shows nested sub-steps.
- ``/api/agent/events`` SSE endpoint — streams ALL events to the Agent Logs
  tab, including recent history on (re)connect.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any, NamedTuple

import structlog

log = structlog.get_logger()


class _Subscriber(NamedTuple):
    """A subscriber queue + the loop it was created on + creation
    timestamp. NamedTuple is hashable as long as its fields are
    hashable. The ``created_at`` field powers the TTL sweep added in
    M3 — subscribers older than ``subscriber_ttl_seconds`` get
    dropped on the next ``subscribe()`` call so dead refs from
    client disconnects don't accumulate indefinitely."""

    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    created_at: float


class ProgressBroker:
    """Fan-out broker for swarm progress events.

    Publishes happen from worker threads; subscribers can live on any
    event loop. Each subscriber's queue is always touched via its own
    loop's ``call_soon_threadsafe``, so publishing is safe regardless of
    which thread/loop the publisher runs on.
    """

    # M3: max age a subscriber can live without being pruned. An
    # hour is generous — SSE clients reconnect every few minutes in
    # practice, so anything older than 1h is almost certainly a
    # stale reference from a disconnected client that didn't reach
    # the finally block cleanly.
    DEFAULT_SUBSCRIBER_TTL_SECONDS: float = 3600.0

    def __init__(
        self,
        *,
        history_size: int = 200,
        queue_size: int = 1000,
        subscriber_ttl_seconds: float | None = None,
    ) -> None:
        self._queue_size = queue_size
        self._subscribers: set[_Subscriber] = set()
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._subscriber_ttl = (
            subscriber_ttl_seconds
            if subscriber_ttl_seconds is not None
            else self.DEFAULT_SUBSCRIBER_TTL_SECONDS
        )

    # ── Publish (thread-safe, loop-agnostic) ─────────────────────────────────

    def publish(self, event: dict[str, Any]) -> None:
        """Fan out an event to all subscribers. Safe to call from any thread
        or loop. Adds a ``timestamp`` field (epoch seconds) if not present."""
        enriched = {"timestamp": time.time(), **event}

        with self._lock:
            self._history.append(enriched)
            targets = list(self._subscribers)

        dead: list[_Subscriber] = []
        for sub in targets:
            try:
                # Each subscriber gets its event delivered on ITS OWN loop —
                # fixes C2: multi-loop deployments and TestClient per-request loops.
                sub.loop.call_soon_threadsafe(self._try_put, sub.queue, enriched)
            except RuntimeError:
                # Loop closed between snapshot and delivery — mark dead so
                # we drop the subscriber and don't keep retrying forever (H1).
                dead.append(sub)

        if dead:
            with self._lock:
                self._subscribers -= set(dead)
            log.debug("broker_pruned_dead_subscribers", count=len(dead))

    @staticmethod
    def _try_put(queue: asyncio.Queue, event: dict[str, Any]) -> None:
        """Runs on the subscriber's event loop; non-blocking put that
        drops the oldest entry on overflow."""
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop-oldest policy: evict one, try again; on second failure give up
            try:
                queue.get_nowait()
                queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    # ── Subscribe / unsubscribe (must be called from an event loop) ──────────

    def subscribe(self, *, include_history: bool = True) -> asyncio.Queue:
        """Create a new subscriber queue bound to the **current** event loop.

        Must be called from the event loop thread that will consume events.
        If ``include_history`` is True, the queue is pre-populated with the
        most recent events.
        """
        # C3 fix: assert loop-thread invariant — this must run on a loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "ProgressBroker.subscribe() must be called from within a "
                "running event loop"
            ) from exc

        now = time.time()
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        sub = _Subscriber(queue=queue, loop=loop, created_at=now)

        with self._lock:
            # M3 fix: sweep stale subscribers on every new subscribe
            # call. Amortises the cleanup cost (no background task
            # needed) and bounds the worst-case accumulation to the
            # interval between subscriptions. A client disconnect
            # that didn't hit the finally block would otherwise leave
            # a stale ref in _subscribers forever, slowly growing
            # publish() iteration cost.
            stale_cutoff = now - self._subscriber_ttl
            stale = {
                s for s in self._subscribers if s.created_at < stale_cutoff
            }
            if stale:
                self._subscribers -= stale
                log.info(
                    "broker_ttl_pruned_stale_subscribers",
                    count=len(stale),
                    ttl_seconds=self._subscriber_ttl,
                )

            if include_history:
                # Safe: we're on the same loop as the queue, put_nowait is OK
                for ev in self._history:
                    try:
                        queue.put_nowait(ev)
                    except asyncio.QueueFull:
                        break
            self._subscribers.add(sub)

        log.debug("broker_subscribed", subscribers=len(self._subscribers))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        with self._lock:
            self._subscribers = {s for s in self._subscribers if s.queue is not queue}
        log.debug("broker_unsubscribed", subscribers=len(self._subscribers))

    # ── Maintenance ──────────────────────────────────────────────────────────

    def clear_history(self) -> None:
        """Drop all events from the history ring buffer.

        Useful between pipeline runs so a newly-connected Agent Logs tab
        doesn't replay events from a previous run.
        """
        with self._lock:
            self._history.clear()

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    @property
    def history_count(self) -> int:
        with self._lock:
            return len(self._history)
