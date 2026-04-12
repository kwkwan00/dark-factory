"""Background metrics writer.

Hot paths (swarm worker threads, LLM clients, orchestrator) call
:meth:`MetricsRecorder.record_*` which enqueues a small tuple and returns
immediately. A single daemon thread drains the queue and writes to
Postgres through :class:`MetricsRepository`.

Design choices:

- **Non-blocking.** If the queue is full (Postgres down or slow), the
  recorder drops the event and logs a warning rather than back-pressuring
  the pipeline. Metric loss is always preferred over pipeline stalls.
- **Fault-tolerant.** Individual write failures are logged and swallowed;
  the worker keeps running.
- **Graceful shutdown.** :meth:`close` poisons the queue and joins the
  worker with a timeout so lifespan shutdown stays bounded.
- **Run-id context.** The recorder keeps a current ``run_id`` so callers
  don't have to thread it through every callsite. The ag-ui bridge sets
  it at the start of a pipeline run and clears it at the end.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import structlog

from dark_factory.metrics.repository import MetricsRepository

log = structlog.get_logger()


# Sentinel used to wake the worker loop on shutdown.
_POISON = object()


class MetricsRecorder:
    """Background writer for :class:`MetricsRepository`."""

    def __init__(
        self,
        repository: MetricsRepository,
        *,
        queue_size: int = 2000,
    ) -> None:
        self._repo = repository
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._running = False
        self._run_id: str | None = None
        self._run_id_lock = threading.Lock()
        self._dropped_count = 0

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker,
            name="metrics-recorder",
            daemon=True,
        )
        self._thread.start()
        log.info("metrics_recorder_started")

    def close(self, timeout: float = 5.0) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._queue.put_nowait(_POISON)
        except queue.Full:
            # Drop an item to make room for the poison pill so the worker
            # can exit promptly.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(_POISON)
            except queue.Full:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info(
            "metrics_recorder_stopped",
            dropped=self._dropped_count,
        )

    # ── Run-id context ───────────────────────────────────────────────────

    def set_run_id(self, run_id: str | None) -> None:
        with self._run_id_lock:
            self._run_id = run_id

    def get_run_id(self) -> str | None:
        with self._run_id_lock:
            return self._run_id

    # ── Public record_* API ──────────────────────────────────────────────

    def record_progress_event(self, event_dict: dict[str, Any]) -> None:
        """Enqueue a raw progress broker event for ingestion."""
        self._enqueue(("progress", event_dict))

    def record_llm_call(self, **fields: Any) -> None:
        self._enqueue(("llm_call", fields))

    def record_pipeline_run_start(self, **fields: Any) -> None:
        self._enqueue(("run_start", fields))

    def record_pipeline_run_end(self, **fields: Any) -> None:
        self._enqueue(("run_end", fields))

    def record_tool_call(self, **fields: Any) -> None:
        self._enqueue(("tool_call", fields))

    def record_agent_stats(self, **fields: Any) -> None:
        self._enqueue(("agent_stats", fields))

    def record_decomposition_stats(self, **fields: Any) -> None:
        self._enqueue(("decomposition", fields))

    def record_memory_operation(self, **fields: Any) -> None:
        self._enqueue(("memory_op", fields))

    def record_incident(self, **fields: Any) -> None:
        self._enqueue(("incident", fields))

    def record_artifact_write(self, **fields: Any) -> None:
        self._enqueue(("artifact", fields))

    def record_background_loop_sample(self, **fields: Any) -> None:
        self._enqueue(("bg_loop", fields))

    def record_swarm_feature_event(self, **fields: Any) -> None:
        self._enqueue(("swarm_feature", fields))

    # ── Worker ───────────────────────────────────────────────────────────

    def _enqueue(self, item: tuple[str, Any]) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped_count += 1
            try:
                from dark_factory.metrics.prometheus import (
                    metrics_events_dropped_by_kind_total,
                    metrics_events_dropped_total,
                )

                metrics_events_dropped_total.inc()
                metrics_events_dropped_by_kind_total.labels(kind=item[0]).inc()
            except Exception:  # pragma: no cover — defensive
                pass
            # H5 fix: log every drop, not just every 100th. Rate-limit
            # only applies to the "totals" summary block appended to
            # the message. Operators need to see drops immediately —
            # a silent metrics outage masked by the old every-100th
            # cadence can run for thousands of events before surfacing.
            log.warning(
                "metrics_queue_full_dropping_event",
                kind=item[0],
                total_dropped=self._dropped_count,
            )

    def _worker(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is _POISON:
                break
            kind, payload = item
            try:
                self._handle(kind, payload)
            except Exception as exc:
                log.warning(
                    "metrics_write_failed",
                    kind=kind,
                    error=str(exc),
                )

    def _handle(self, kind: str, payload: Any) -> None:
        if kind == "progress":
            self._repo.ingest_progress_event(
                event_dict=payload,
                run_id=self.get_run_id(),
            )
        elif kind == "llm_call":
            payload.setdefault("run_id", self.get_run_id())
            self._repo.record_llm_call(**payload)
        elif kind == "run_start":
            self._repo.record_pipeline_run_start(**payload)
        elif kind == "run_end":
            self._repo.record_pipeline_run_end(**payload)
        elif kind == "tool_call":
            payload.setdefault("run_id", self.get_run_id())
            self._repo.record_tool_call(**payload)
        elif kind == "agent_stats":
            payload.setdefault("run_id", self.get_run_id())
            self._repo.record_agent_stats(**payload)
        elif kind == "decomposition":
            payload.setdefault("run_id", self.get_run_id())
            self._repo.record_decomposition_stats(**payload)
        elif kind == "memory_op":
            payload.setdefault("run_id", self.get_run_id())
            self._repo.record_memory_operation(**payload)
        elif kind == "incident":
            payload.setdefault("run_id", self.get_run_id())
            self._repo.record_incident(**payload)
        elif kind == "artifact":
            payload.setdefault("run_id", self.get_run_id())
            self._repo.record_artifact_write(**payload)
        elif kind == "bg_loop":
            self._repo.record_background_loop_sample(**payload)
        elif kind == "swarm_feature":
            payload.setdefault("run_id", self.get_run_id())
            self._repo.record_swarm_feature_event(**payload)

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def queue_depth(self) -> int:
        return self._queue.qsize()


def build_recorder_from_settings(settings) -> "tuple[MetricsRecorder | None, object | None]":
    """Best-effort factory for use in ``api/app.py`` lifespan.

    Returns ``(recorder, client)``. Either can be ``None`` if Postgres is
    disabled or failed to initialise — callers must tolerate both. The
    client is returned so the lifespan can close it on shutdown.
    """
    cfg = settings.postgres
    if not cfg.enabled:
        log.info("metrics_disabled", reason="postgres.enabled=false")
        return None, None

    try:
        from dark_factory.metrics import PostgresClient, ensure_schema

        client = PostgresClient(cfg)
        ensure_schema(client)
        recorder = MetricsRecorder(
            MetricsRepository(client),
            queue_size=cfg.recorder_queue_size,
        )
        recorder.start()
        log.info("metrics_enabled", url=cfg.url)
        return recorder, client
    except Exception as exc:
        log.warning("metrics_init_failed", error=str(exc), url=cfg.url)
        return None, None
