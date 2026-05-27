"""Cross-debate coordination bus.

Concurrent per-requirement debates run in worker threads. Without
coordination, two debates that hit the same architectural concern
re-derive the same finding independently — wasted tokens and risk
of inconsistent verdicts on related requirements.

The bus is an in-process, thread-safe pub/sub keyed on
``refinery_run_id``. Each worker pushes structured findings at the
end of its rounds (critic blockers + judge tradeoffs); other workers
read what their concurrent siblings have noticed and the prompt
formatter renders a "concurrent debates noticed:" section into the
next round's critic prompts. Bounded buffer per run; cleared at run
end.

Scope is deliberately narrow:

- One process. The bus does NOT cross worker processes or hosts —
  if multiple SSE consumers run in different processes, each has
  its own bus. Cross-process coordination would need Redis / NATS
  / similar; not in V1.
- Idempotent reads. Workers read the bus by event id; duplicate
  reads return the same payload. The bus never deletes entries
  during a run; ``finish_run`` clears them at the end.
- Bounded by configuration. ``max_findings_per_run`` caps the
  buffer so a runaway producer can't OOM the orchestrator.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from dark_factory.log import trace_methods

log = structlog.get_logger()


@dataclass
class CrossDebateFinding:
    """One observation a debate emits for its siblings.

    The producer is the requirement_id whose debate observed the
    finding; the kind tells the formatter how to render it; the body
    carries the structured payload. ``id`` is monotonically unique
    within a run so consumers can dedup across reads."""

    id: str
    refinery_run_id: str
    requirement_id: str
    kind: str  # "critic_blocker" | "judge_tradeoff" | "constraint" | …
    body: dict[str, Any]
    round_number: int = 0
    created_at: float = field(default_factory=time.time)


_DEFAULT_TTL_SECONDS = 60 * 60  # one hour
"""Per-run buffers older than this are evicted on the next post.

Covers the failure / cancellation paths where ``finish_run`` never
fires — bounded TTL stops long-lived processes from accumulating
abandoned per-run state. Tuned for the typical refinery run length
(seconds-to-minutes); operators with longer-running runs can pass a
larger value at construction.
"""


@trace_methods
class CrossDebateBus:
    """Thread-safe pub/sub keyed on refinery_run_id.

    Each ``post()`` adds a finding to the run's buffer. Each
    ``read_for(req_id)`` returns the findings posted by *other*
    requirements in the same run, optionally filtered by kind. The
    bus never cross-pollinates between runs.
    """

    def __init__(
        self,
        *,
        max_findings_per_run: int = 200,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._lock = threading.RLock()
        self._buffers: dict[str, list[CrossDebateFinding]] = {}
        self._last_active: dict[str, float] = {}
        self._max = max_findings_per_run
        self._ttl = ttl_seconds

    # ── Producer ─────────────────────────────────────────────────────

    def post(
        self,
        *,
        refinery_run_id: str,
        requirement_id: str,
        kind: str,
        body: dict[str, Any],
        round_number: int = 0,
    ) -> str:
        """Append one finding. Returns the assigned id so the producer
        can later filter its own posts back out."""

        finding = CrossDebateFinding(
            id=uuid.uuid4().hex,
            refinery_run_id=refinery_run_id,
            requirement_id=requirement_id,
            kind=kind,
            body=body,
            round_number=round_number,
        )
        with self._lock:
            self._evict_aged_out_locked()
            buf = self._buffers.setdefault(refinery_run_id, [])
            buf.append(finding)
            # Drop oldest entries when the cap is exceeded so we never
            # OOM. ``-self._max:`` slice is O(N) but the cap is small.
            if len(buf) > self._max:
                self._buffers[refinery_run_id] = buf[-self._max:]
            self._last_active[refinery_run_id] = finding.created_at
        return finding.id

    def _evict_aged_out_locked(self) -> None:
        """Drop per-run buffers whose last activity exceeded the TTL.

        Caller MUST hold ``self._lock``. This runs from every public
        entrypoint that takes the lock (``post`` / ``read_for`` /
        ``stats``) so a process whose only activity is a slow stream
        of reads still evicts stale buffers — not just a busy poster
        process. The bound is the TTL, not "forever"."""

        if not self._last_active:
            return
        cutoff = time.time() - self._ttl
        stale = [
            run_id for run_id, ts in self._last_active.items() if ts < cutoff
        ]
        for run_id in stale:
            self._buffers.pop(run_id, None)
            self._last_active.pop(run_id, None)

    # ── Consumer ─────────────────────────────────────────────────────

    def read_for(
        self,
        *,
        refinery_run_id: str,
        requirement_id: str,
        kinds: list[str] | None = None,
        limit: int = 20,
    ) -> list[CrossDebateFinding]:
        """Return findings posted by debates *other than*
        ``requirement_id`` for this run.

        The list is newest-first and capped at ``limit``. Filtered to
        ``kinds`` when provided. Returns an empty list when the run
        has no posts yet."""

        with self._lock:
            self._evict_aged_out_locked()
            buf = list(self._buffers.get(refinery_run_id, []))
        out: list[CrossDebateFinding] = []
        for f in reversed(buf):
            if f.requirement_id == requirement_id:
                continue
            if kinds is not None and f.kind not in kinds:
                continue
            out.append(f)
            if len(out) >= limit:
                break
        return out

    # ── Lifecycle ────────────────────────────────────────────────────

    def finish_run(self, refinery_run_id: str) -> None:
        """Clear the buffer for one run. Called by the orchestrator at
        Phase 4 on the success path. Cancelled / errored runs that
        never reach this call get evicted by the TTL on the next
        post — see ``_evict_aged_out_locked``."""

        with self._lock:
            self._buffers.pop(refinery_run_id, None)
            self._last_active.pop(refinery_run_id, None)

    def stats(self, refinery_run_id: str) -> dict[str, int]:
        """Operator-facing snapshot — counts per kind."""

        with self._lock:
            self._evict_aged_out_locked()
            buf = list(self._buffers.get(refinery_run_id, []))
        out: dict[str, int] = {}
        for f in buf:
            out[f.kind] = out.get(f.kind, 0) + 1
        out["total"] = len(buf)
        return out


# Process-global bus instance. The orchestrator passes a reference to
# each worker so all concurrent debates share the same buffer for
# their run. Tests construct their own to avoid bleed.
_GLOBAL_BUS = CrossDebateBus()


def get_global_bus() -> CrossDebateBus:
    """Return the process-shared bus instance."""

    return _GLOBAL_BUS


__all__ = [
    "CrossDebateBus",
    "CrossDebateFinding",
    "get_global_bus",
]
