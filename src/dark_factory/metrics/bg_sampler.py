"""Periodic BackgroundLoop sampler.

A tiny daemon thread that polls :class:`BackgroundLoop` every N seconds and
writes a row to ``background_loop_samples`` via the metrics recorder. Starts
at app lifespan startup, stops at shutdown.
"""

from __future__ import annotations

import threading
import time

import structlog

log = structlog.get_logger()


class BackgroundLoopSampler:
    """Periodic sampler thread."""

    def __init__(self, interval_seconds: float = 10.0) -> None:
        self._interval = max(1.0, interval_seconds)
        self._thread: threading.Thread | None = None
        self._running = False
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="metrics-bg-sampler",
            daemon=True,
        )
        self._thread.start()
        log.info("bg_loop_sampler_started", interval_seconds=self._interval)

    def stop(self, timeout: float = 5.0) -> None:
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info("bg_loop_sampler_stopped")

    def _loop(self) -> None:
        from dark_factory.agents.background_loop import BackgroundLoop
        from dark_factory.metrics.helpers import record_background_loop_sample

        while self._running:
            try:
                bg = BackgroundLoop.get()
                active, pending, completed = bg.task_counts()
                record_background_loop_sample(
                    active_task_count=active,
                    pending_task_count=pending,
                    completed_task_count=completed,
                    loop_restarts=BackgroundLoop.restart_count(),
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("bg_loop_sample_failed", error=str(exc))

            # Progress broker gauges — Prometheus only.
            try:
                from dark_factory.agents import tools as _tools_mod
                from dark_factory.metrics.prometheus import observe_progress_broker

                broker = _tools_mod._progress_broker
                if broker is not None:
                    observe_progress_broker(
                        subscribers=broker.subscriber_count,
                        history_size=broker.history_count,
                    )
            except Exception:  # pragma: no cover — defensive
                pass

            # Metrics recorder queue gauges — Prometheus only.
            try:
                from dark_factory.agents import tools as _tools_mod
                from dark_factory.metrics.prometheus import observe_metrics_recorder

                recorder = _tools_mod._metrics_recorder
                if recorder is not None:
                    observe_metrics_recorder(
                        queue_depth=recorder.queue_depth(),
                        dropped_delta=0,  # counter already maintained by recorder
                    )
            except Exception:  # pragma: no cover — defensive
                pass

            # L5 fix: Postgres pool gauges. Emitted every sampler tick
            # so dashboards can show pool exhaustion trends before
            # they cause pipeline stalls. Walks the metrics_client
            # held by the recorder — no direct access to the pool
            # from here.
            try:
                from dark_factory.agents import tools as _tools_mod
                from dark_factory.metrics.prometheus import observe_postgres_pool

                recorder = _tools_mod._metrics_recorder
                client = getattr(recorder, "_client", None) if recorder else None
                if client is not None and hasattr(client, "pool_stats"):
                    observe_postgres_pool(**client.pool_stats())
            except Exception:  # pragma: no cover — defensive
                pass

            # Interruptible sleep
            self._stop_event.wait(timeout=self._interval)
            if self._stop_event.is_set():
                break
