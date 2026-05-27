"""Requirement-level resume support for the refinery SSE pipeline.

The orchestrator records run lifecycle + per-debate completions so an
interrupted run can be picked up from where it left off without
re-running already-completed debates. Both the registry writes and
the resume reads degrade gracefully when Postgres is disabled — the
feature simply becomes a no-op.

Resume granularity is **per requirement**, not per debate round. A
debate that was mid-flight when the worker died restarts from scratch;
debates that completed before the failure replay from cache. This is
the typical failure mode (5 of 20 requirements completed, then crash)
and gives the largest practical win for the smallest implementation
surface. Mid-debate resume (LangGraph checkpointer) is V2.
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()


_RUN_STATUSES = {"in_progress", "completed", "cancelled", "failed"}


class ResumeRegistry:
    """Thin adapter over ``RefineryMetricsRepository`` that no-ops every
    call when the underlying Postgres client is unavailable.

    The orchestrator should construct one per request from
    ``request.app.state.metrics_client``. Tests construct one with
    ``client=None`` to skip persistence entirely.
    """

    def __init__(self, metrics_client: Any) -> None:
        self._repo = None
        if metrics_client is None:
            return
        try:
            from dark_factory.metrics.refinery_repository import (
                RefineryMetricsRepository,
            )

            self._repo = RefineryMetricsRepository(metrics_client)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("refinery_resume_registry_init_failed", error=str(exc))
            self._repo = None

    @property
    def enabled(self) -> bool:
        return self._repo is not None

    # ── Run lifecycle ─────────────────────────────────────────────────────

    def start(
        self,
        *,
        refinery_run_id: str,
        source_mode: str,
        source_run_id: str | None,
        input_snapshot: dict[str, Any],
    ) -> None:
        """Create the run row + persist the input payload so a resume
        can skip Phase 1. Idempotent on ``refinery_run_id``."""

        if self._repo is None:
            return
        try:
            requirements_count = len(input_snapshot.get("requirements") or [])
            self._repo.record_run_start(
                refinery_run_id=refinery_run_id,
                source_mode=source_mode,
                source_run_id=source_run_id,
                requirements_count=requirements_count,
                settings_snapshot={},
            )
            self._repo.record_input_snapshot(
                refinery_run_id=refinery_run_id,
                input_snapshot=input_snapshot,
            )
            self._repo.record_run_status(
                refinery_run_id=refinery_run_id,
                status="in_progress",
            )
        except Exception:  # pragma: no cover — defensive
            log.exception("refinery_resume_start_failed",
                          refinery_run_id=refinery_run_id)

    def mark_status(self, refinery_run_id: str, status: str) -> None:
        """Set the run's lifecycle status. Status must be one of
        ``in_progress`` / ``completed`` / ``cancelled`` / ``failed``."""

        if self._repo is None or status not in _RUN_STATUSES:
            return
        try:
            self._repo.record_run_status(
                refinery_run_id=refinery_run_id, status=status,
            )
        except Exception:  # pragma: no cover — defensive
            log.exception("refinery_resume_status_failed",
                          refinery_run_id=refinery_run_id, status=status)

    def try_claim_in_progress(self, refinery_run_id: str) -> bool:
        """Atomic single-winner transition to ``in_progress``. Returns
        True iff this caller successfully claimed the run; concurrent
        claimers see False and should refuse to proceed.

        When Postgres is unwired this returns True (no contention to
        worry about) so single-instance dev setups still work."""

        if self._repo is None:
            return True
        try:
            return self._repo.try_claim_in_progress(refinery_run_id)
        except Exception:  # pragma: no cover — defensive
            log.exception("refinery_resume_claim_failed",
                          refinery_run_id=refinery_run_id)
            return True

    def cache_debate(
        self,
        *,
        refinery_run_id: str,
        requirement_id: str,
        convergence_status: str,
        refined_payload: dict[str, Any],
        suggested_memories: list[dict[str, Any]] | None = None,
    ) -> None:
        """Cache one completed debate's output so resume can replay it
        without re-invoking the panel."""

        if self._repo is None:
            return
        try:
            self._repo.record_debate_completion(
                refinery_run_id=refinery_run_id,
                requirement_id=requirement_id,
                convergence_status=convergence_status,
                refined_payload=refined_payload,
                suggested_memories=suggested_memories or [],
            )
        except Exception:  # pragma: no cover — defensive
            log.exception(
                "refinery_resume_cache_debate_failed",
                refinery_run_id=refinery_run_id,
                requirement_id=requirement_id,
            )

    # ── Resume reads ──────────────────────────────────────────────────────

    def load_run(self, refinery_run_id: str) -> dict[str, Any] | None:
        """Return the run's resume-relevant state, or ``None`` when
        Postgres is disabled or the run is unknown."""

        if self._repo is None:
            return None
        try:
            return self._repo.load_run_for_resume(refinery_run_id)
        except Exception:  # pragma: no cover — defensive
            log.exception("refinery_resume_load_run_failed",
                          refinery_run_id=refinery_run_id)
            return None

    def load_completed(
        self, refinery_run_id: str,
    ) -> dict[str, dict[str, Any]]:
        """Return ``{req_id: {convergence_status, refined, suggested_memories,
        completed_at}}`` for every cached completion of the run."""

        if self._repo is None:
            return {}
        try:
            return self._repo.load_completed_debates(refinery_run_id)
        except Exception:  # pragma: no cover — defensive
            log.exception("refinery_resume_load_completed_failed",
                          refinery_run_id=refinery_run_id)
            return {}


__all__ = ["ResumeRegistry"]
