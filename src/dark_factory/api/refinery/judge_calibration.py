"""Judge confidence calibration — bounded threshold multiplier.

Mirrors the role-weighting and provider-learning loops, but the
signal is the **Judge's own predictions** vs. operator follow-through:

- Judge said "pass" (high overall score) → operator applied:
  ``high_applied`` — calibrated, no adjustment.
- Judge said "pass" → operator dismissed:
  ``high_dismissed`` — Judge was overconfident, **raise** the
  threshold so future debates clear a higher bar before passing.
- Judge said "fail" (low overall score) → operator applied:
  ``low_applied`` — Judge was underconfident, **lower** the threshold
  so debates that would have stalled converge sooner.
- Judge said "fail" → operator dismissed:
  ``low_dismissed`` — calibrated.

The result is a **threshold multiplier** in a tight band [0.85, 1.15],
designed to nudge convergence behaviour without rewriting any
prompts. ``effective_threshold = base_threshold * multiplier``:

- multiplier > 1.0 → harder to pass (Judge has been overconfident)
- multiplier < 1.0 → easier to pass (Judge has been underconfident)
- multiplier = 1.0 → no signal yet, identity behaviour

Bounds are tighter than role-weighting (0.60–1.40) because moving
the convergence threshold has cascading effects on rounds-per-debate
and cost. A small nudge is enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()


_MIN_TOTAL_FOR_LEARNING = 10
"""Below this many decisions, ignore the signal — too little
operator follow-through to trust the rate."""

_MISCALIBRATION_TRIGGER = 0.30
"""Above this miscalibration ratio, nudge the threshold. Computed as
(high_dismissed + low_applied) / total — both arms of error in one
number."""

_DEFAULT_MULTIPLIER = 1.0
_MIN_MULTIPLIER = 0.85
_MAX_MULTIPLIER = 1.15
_STEP = 0.05


@dataclass
class JudgeCalibration:
    """Effective threshold multiplier + audit trail."""

    multiplier: float
    high_applied: float
    high_dismissed: float
    low_applied: float
    low_dismissed: float
    total: float
    miscalibration_rate: float
    direction: str
    """``overconfident`` | ``underconfident`` | ``calibrated`` | ``insufficient_signal``"""
    reason: str


def adjust_judge_threshold(
    stats: dict[str, Any] | None,
) -> JudgeCalibration:
    """Convert raw decision counts into a bounded threshold multiplier.

    ``stats`` is the dict returned by
    :meth:`RefineryMetricsRepository.compute_judge_calibration_stats`.
    Empty / None → identity multiplier with ``insufficient_signal``.
    """

    high_applied = float((stats or {}).get("high_applied") or 0.0)
    high_dismissed = float((stats or {}).get("high_dismissed") or 0.0)
    low_applied = float((stats or {}).get("low_applied") or 0.0)
    low_dismissed = float((stats or {}).get("low_dismissed") or 0.0)
    total = high_applied + high_dismissed + low_applied + low_dismissed
    miscal = ((high_dismissed + low_applied) / total) if total else 0.0

    def _build(multiplier: float, direction: str, reason: str) -> JudgeCalibration:
        return JudgeCalibration(
            multiplier=multiplier,
            high_applied=high_applied, high_dismissed=high_dismissed,
            low_applied=low_applied, low_dismissed=low_dismissed,
            total=total, miscalibration_rate=miscal,
            direction=direction, reason=reason,
        )

    if not stats:
        return _build(_DEFAULT_MULTIPLIER, "insufficient_signal", "no_stats")
    if total < _MIN_TOTAL_FOR_LEARNING:
        return _build(_DEFAULT_MULTIPLIER, "insufficient_signal", "below_min_total")
    if miscal < _MISCALIBRATION_TRIGGER:
        return _build(_DEFAULT_MULTIPLIER, "calibrated", "within_tolerance")

    # Pick the direction by which error arm dominates. Tied → no move.
    if high_dismissed > low_applied:
        # Judge passes things operators end up dismissing → too lenient.
        return _build(
            min(_MAX_MULTIPLIER, _DEFAULT_MULTIPLIER + _STEP),
            "overconfident", "high_dismissed_dominates",
        )
    if low_applied > high_dismissed:
        # Judge fails things operators end up applying → too strict.
        return _build(
            max(_MIN_MULTIPLIER, _DEFAULT_MULTIPLIER - _STEP),
            "underconfident", "low_applied_dominates",
        )
    return _build(_DEFAULT_MULTIPLIER, "calibrated", "balanced_error_arms")


def load_judge_calibration(
    *,
    metrics_client: Any | None,
    window_days: int = 30,
    score_split: float = 0.8,
    stats: dict[str, Any] | None = None,
) -> JudgeCalibration:
    """Top-level entry — fetch stats from Postgres (or take precomputed
    ones for an endpoint that already has them) and convert to a
    multiplier.

    Returns the identity-default ``JudgeCalibration`` on any failure
    path. Callers treat ``multiplier == 1.0`` as a no-op so the system
    never hard-fails because calibration can't load.
    """

    if stats is None:
        if metrics_client is None:
            return adjust_judge_threshold(None)
        try:
            from dark_factory.metrics.refinery_repository import (
                RefineryMetricsRepository,
            )
            repo = RefineryMetricsRepository(metrics_client)
            stats = repo.compute_judge_calibration_stats(
                window_days=window_days, score_split=score_split,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_judge_calibration_load_failed", error=str(exc),
            )
            return adjust_judge_threshold(None)

    return adjust_judge_threshold(stats)


__all__ = [
    "JudgeCalibration",
    "adjust_judge_threshold",
    "load_judge_calibration",
]
