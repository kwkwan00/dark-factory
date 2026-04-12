"""File watcher for auto-triggering pipeline on spec changes."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

import structlog
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

log = structlog.get_logger()


@dataclass
class WatchEvent:
    """A detected file change event."""

    path: str
    event_type: str  # "created" | "modified" | "deleted"
    timestamp: float = field(default_factory=time.time)


class _DebouncedHandler(FileSystemEventHandler):
    """Watchdog handler with debounce: waits for quiet period before firing callback."""

    def __init__(self, debounce_seconds: int, event_queue: Queue) -> None:
        self.debounce_seconds = debounce_seconds
        self.event_queue = event_queue
        # M10 fix: accumulate {path: event_type} of all events in the debounce window
        self._pending: dict[str, str] = {}
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def _on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # Only watch spec-related files
        if not event.src_path.endswith((".md", ".json", ".txt", ".yaml")):
            return

        with self._lock:
            # Keep the latest event type per path (created → modified overrides)
            self._pending[event.src_path] = event.event_type

            # Reset debounce timer
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            for path, event_type in self._pending.items():
                self.event_queue.put(WatchEvent(path=path, event_type=event_type))
                log.info("watch_event", path=path, type=event_type)
            self._pending.clear()

    def on_created(self, event: FileSystemEvent) -> None:
        self._on_any_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._on_any_event(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._on_any_event(event)


class FileWatcher:
    """Watches directories for file changes and posts events to a queue."""

    def __init__(
        self,
        paths: list[str],
        debounce_seconds: int = 5,
    ) -> None:
        self.paths = paths
        self.debounce_seconds = debounce_seconds
        self.event_queue: Queue = Queue()
        self.event_log: deque[WatchEvent] = deque(maxlen=50)
        self._observer: Observer | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_event(self) -> WatchEvent | None:
        return self.event_log[-1] if self.event_log else None

    def start(self) -> None:
        """Start watching all configured paths."""
        if self._running:
            return

        self._observer = Observer()
        handler = _DebouncedHandler(self.debounce_seconds, self.event_queue)

        for path_str in self.paths:
            path = Path(path_str)
            if path.is_dir():
                self._observer.schedule(handler, str(path), recursive=True)
                log.info("watcher_scheduled", path=str(path))
            else:
                log.warning("watcher_path_not_found", path=str(path))

        self._observer.daemon = True
        self._observer.start()
        self._running = True
        log.info("watcher_started", paths=self.paths)

    def stop(self) -> None:
        """Stop watching."""
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
            self._running = False
            log.info("watcher_stopped")

    def drain_events(self) -> list[WatchEvent]:
        """Drain all pending events from the queue."""
        import queue as _queue

        events: list[WatchEvent] = []
        while True:
            try:
                event = self.event_queue.get_nowait()
            except _queue.Empty:
                break
            events.append(event)
            self.event_log.append(event)
        return events
