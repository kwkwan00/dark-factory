"""Adaptive max_rounds per requirement archetype — bounded multiplier.

Mirrors the role-weighting and judge-calibration loops, but the signal
is **how many rounds historically-similar debates needed to converge**.

A requirement archetype is the bucket
``(priority, source_mode, primary_tag)`` — coarse enough that low-volume
deployments still hit the learning threshold, and fine enough that
"high-priority security" and "low-priority internal-tooling" don't
share a knob.

Signal interpretation per archetype:

- **``avg_rounds_converged << base``** — converged debates in this
  archetype regularly finish well under the cap. Reduce ``max_rounds``
  so future debates don't burn rounds re-confirming consensus.
- **``short_circuit_rate >> floor``** — debates in this archetype hit
  the cap without converging often. Bump ``max_rounds`` so the panel
  has more room to resolve its disagreements before reconcile fires.
- **Otherwise** — identity (no change).

The result is a **bounded effective max_rounds** scaled from the
configured base. Bounds are tight (``[2, base+2]`` clamped to a
hard ceiling of 6) because rounds compose with critic-fanout cost
and recursion-limit headroom downstream.

Identity-default everywhere: missing data, missing client, ambiguous
signal → return the configured base unchanged. Bounds are enforced
even when signal is strong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()


_MIN_TOTAL_FOR_LEARNING = 8
"""Below this many debates in the archetype window, ignore the signal."""

_REDUCE_RATIO = 0.55
"""When mean converged rounds is below this fraction of the base, the
archetype is over-budgeted — reduce by 1 round."""

_BUMP_SHORT_CIRCUIT_RATE = 0.30
"""When more than this fraction of debates short-circuit, the archetype
is under-budgeted — bump by 1 round (or 2 when the rate is very high)."""

_BUMP_HIGH_SHORT_CIRCUIT_RATE = 0.55
"""Above this rate, bump by 2 rounds — the panel is genuinely starved."""

_HARD_CEILING = 6
"""Absolute ceiling regardless of signal. Each extra round multiplies
critic-fanout cost; uncapped growth defeats the bounded-multiplier
discipline."""

_HARD_FLOOR = 2
"""Below 2 rounds the adversarial loop has nothing to iterate on —
generator + one critique pass is the minimum viable debate."""


@dataclass(frozen=True)
class ArchetypeRounds:
    """Effective max_rounds for one archetype + audit trail."""

    archetype_key: str
    effective_max_rounds: int
    base_max_rounds: int
    total: int
    converged: int
    short_circuited: int
    short_circuit_rate: float
    avg_rounds_converged: float | None
    direction: str
    """``reduce`` | ``bump_one`` | ``bump_two`` | ``identity`` | ``insufficient_signal``"""
    reason: str


def archetype_key_for(
    *,
    priority: str | None,
    source_mode: str | None,
    tags: list[str] | None,
) -> str:
    """Canonical bucket string. Stable across runs so aggregator GROUP-BYs
    align with the lookup at injection time."""

    p = (priority or "unknown").strip().lower() or "unknown"
    s = (source_mode or "unknown").strip().lower() or "unknown"
    primary_tag = "untagged"
    if tags:
        first = next((t for t in tags if isinstance(t, str) and t.strip()), None)
        if first:
            primary_tag = first.strip().lower()
    return f"{p}|{s}|{primary_tag}"


def _classify(
    *,
    base: int,
    total: int,
    converged: int,
    short_circuited: int,
    avg_rounds_converged: float | None,
) -> tuple[int, str, str]:
    """Pure decision: ``(effective_max_rounds, direction, reason)``."""

    if total < _MIN_TOTAL_FOR_LEARNING:
        return base, "insufficient_signal", "below_min_total"

    sc_rate = (short_circuited / total) if total else 0.0

    # Bump direction wins over reduce when both signals fire — a starved
    # panel needs more room before any token-saving optimisations apply.
    if sc_rate >= _BUMP_HIGH_SHORT_CIRCUIT_RATE:
        return (
            min(_HARD_CEILING, base + 2),
            "bump_two",
            "high_short_circuit_rate",
        )
    if sc_rate >= _BUMP_SHORT_CIRCUIT_RATE:
        return (
            min(_HARD_CEILING, base + 1),
            "bump_one",
            "elevated_short_circuit_rate",
        )

    # Reduce only when we have a meaningful mean to compare against and
    # at least a handful of converged debates to draw it from.
    if (
        avg_rounds_converged is not None
        and converged >= _MIN_TOTAL_FOR_LEARNING
        and avg_rounds_converged <= base * _REDUCE_RATIO
    ):
        return (
            max(_HARD_FLOOR, base - 1),
            "reduce",
            "low_mean_rounds_to_converge",
        )

    return base, "identity", "within_tolerance"


def adjust_archetype_rounds(
    *,
    archetype_stats: list[dict[str, Any]],
    base_max_rounds: int,
) -> dict[str, ArchetypeRounds]:
    """Convert raw archetype stats into per-bucket effective max_rounds.

    Buckets absent from input get no entry in the result; callers that
    want identity behaviour for missing buckets use
    :func:`effective_max_rounds_for`.
    """

    out: dict[str, ArchetypeRounds] = {}
    for row in archetype_stats or []:
        try:
            priority = row.get("priority")
            source_mode = row.get("source_mode")
            primary_tag = row.get("primary_tag")
            total = int(row.get("total") or 0)
            converged = int(row.get("converged") or 0)
            short_c = int(row.get("short_circuited") or 0)
            avg_r_raw = row.get("avg_rounds_converged")
            avg_r = float(avg_r_raw) if avg_r_raw is not None else None
        except (TypeError, ValueError):
            log.warning("refinery_archetype_rounds_bad_row", row=row)
            continue

        key = archetype_key_for(
            priority=priority,
            source_mode=source_mode,
            tags=[primary_tag] if primary_tag else None,
        )
        eff, direction, reason = _classify(
            base=base_max_rounds,
            total=total,
            converged=converged,
            short_circuited=short_c,
            avg_rounds_converged=avg_r,
        )
        out[key] = ArchetypeRounds(
            archetype_key=key,
            effective_max_rounds=eff,
            base_max_rounds=base_max_rounds,
            total=total,
            converged=converged,
            short_circuited=short_c,
            short_circuit_rate=(short_c / total) if total else 0.0,
            avg_rounds_converged=avg_r,
            direction=direction,
            reason=reason,
        )
    return out


def effective_max_rounds_for(
    *,
    priority: str | None,
    source_mode: str | None,
    tags: list[str] | None,
    archetypes: dict[str, ArchetypeRounds],
    base_max_rounds: int,
) -> int:
    """Identity-default lookup. Unknown archetype → ``base_max_rounds``."""

    key = archetype_key_for(
        priority=priority, source_mode=source_mode, tags=tags,
    )
    entry = archetypes.get(key)
    return entry.effective_max_rounds if entry is not None else base_max_rounds


def load_archetype_rounds(
    *,
    metrics_client: Any | None,
    base_max_rounds: int,
    window_days: int = 30,
    archetype_stats: list[dict[str, Any]] | None = None,
) -> dict[str, ArchetypeRounds]:
    """Top-level entry — fetch stats from Postgres (or take precomputed
    ones for endpoints that already have them) and convert to per-
    archetype effective max_rounds.

    Empty dict on any failure path; callers treat empty as "every
    archetype uses ``base_max_rounds``" so the system never hard-fails
    because the loop can't load.
    """

    if archetype_stats is None:
        if metrics_client is None:
            return {}
        try:
            from dark_factory.metrics.refinery_repository import (
                RefineryMetricsRepository,
            )
            repo = RefineryMetricsRepository(metrics_client)
            archetype_stats = repo.compute_archetype_round_stats(
                window_days=window_days,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_archetype_rounds_load_failed", error=str(exc),
            )
            return {}

    return adjust_archetype_rounds(
        archetype_stats=archetype_stats,
        base_max_rounds=base_max_rounds,
    )


__all__ = [
    "ArchetypeRounds",
    "adjust_archetype_rounds",
    "archetype_key_for",
    "effective_max_rounds_for",
    "load_archetype_rounds",
]
