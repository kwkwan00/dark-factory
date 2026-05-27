"""Provider learning loop — adjusts per-tier trust weights based on
the recent contribution rates of the providers feeding each tier.

The hypothesis: when a provider keeps returning sources that the
Editor's cross-tier confidence test rejects (low ``propagated`` rate),
its tier weight should drift downward so future Editor decisions
reflect the empirical signal rather than the static default. When a
provider keeps producing sources that survive into ValidatedInsights,
its tier weight drifts upward.

The adjustment is bounded and reversible — operators can always
override via ``refinery_research_tier_trust_weights`` config, and the
adjustment never crosses zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from dark_factory.api.refinery.contracts import SourceTier

log = structlog.get_logger()


# Default learning hyperparameters. Tuned for slow drift — a single
# poor run shouldn't tank a provider's weight, but a sustained pattern
# of T5 noise should reduce its leverage on confidence scoring.
_LOW_RATE_FLOOR = 0.10
"""Below this contribution rate, the provider gets a soft demotion."""

_HIGH_RATE_CEILING = 0.50
"""Above this contribution rate, the provider gets a soft boost."""

_STEP = 0.05
"""How far each adjustment moves the trust weight per refresh."""

_MIN_CALLS_FOR_LEARNING = 5
"""Below this many calls in the window, ignore the provider — too
little signal to trust the rate."""

_BOUNDS_PER_TIER: dict[SourceTier, tuple[float, float]] = {
    SourceTier.T0_STRUCTURED: (0.90, 1.00),
    SourceTier.T1_INTERNAL: (0.85, 0.99),
    SourceTier.T2_OBSERVABILITY: (0.70, 0.95),
    SourceTier.T3_OFFICIAL: (0.60, 0.90),
    SourceTier.T4_ACADEMIC: (0.40, 0.75),
    SourceTier.T5_WEB: (0.10, 0.50),
}
"""Per-tier (min, max) bounds. Adjustments can't push a tier outside
its envelope so a runaway loop can't degrade a tier into uselessness
or boost an unreliable tier above a reliable one."""


@dataclass
class TierAdjustment:
    """The runtime trust weight to apply to one tier on the next call,
    plus the audit trail (calls / contribution rate) for the operator
    UI."""

    tier: int
    base_weight: float
    adjusted_weight: float
    contribution_rate: float
    calls: int
    propagated: int
    reason: str


def adjust_trust_weights(
    *,
    base_weights: dict[int, float],
    provider_stats: list[dict[str, Any]],
) -> dict[int, TierAdjustment]:
    """Compute per-tier trust adjustments.

    Aggregates the per-(tier, provider) stats up to the tier level —
    a tier's trust adjustment reflects the combined contribution
    rate of every provider feeding it. The default adjustment is
    "no change" when stats are missing or below the
    ``_MIN_CALLS_FOR_LEARNING`` threshold.
    """

    by_tier: dict[int, list[dict[str, Any]]] = {}
    for row in provider_stats:
        by_tier.setdefault(row["tier"], []).append(row)

    out: dict[int, TierAdjustment] = {}
    for tier_val, base in base_weights.items():
        stats = by_tier.get(tier_val, [])
        calls = sum(s["calls"] for s in stats)
        propagated = sum(s["propagated"] for s in stats)
        rate = (propagated / calls) if calls else 0.0

        adjusted = float(base)
        reason = "no change"

        if calls < _MIN_CALLS_FOR_LEARNING:
            reason = f"insufficient signal ({calls} calls < {_MIN_CALLS_FOR_LEARNING})"
        elif rate < _LOW_RATE_FLOOR:
            adjusted = base - _STEP
            reason = (
                f"contribution rate {rate:.1%} below floor {_LOW_RATE_FLOOR:.0%} "
                f"— soft demotion"
            )
        elif rate > _HIGH_RATE_CEILING:
            adjusted = base + _STEP
            reason = (
                f"contribution rate {rate:.1%} above ceiling {_HIGH_RATE_CEILING:.0%} "
                f"— soft boost"
            )

        # Apply per-tier bounds so a learning loop can't drift outside
        # the envelope the architect set.
        try:
            tier_enum = SourceTier(tier_val)
            lo, hi = _BOUNDS_PER_TIER[tier_enum]
        except (ValueError, KeyError):
            lo, hi = 0.0, 1.0
        adjusted = max(lo, min(hi, adjusted))

        out[tier_val] = TierAdjustment(
            tier=tier_val,
            base_weight=float(base),
            adjusted_weight=adjusted,
            contribution_rate=rate,
            calls=calls,
            propagated=propagated,
            reason=reason,
        )

    return out


def load_adjusted_trust_weights(
    *,
    metrics_client: Any,
    base_weights: dict[int, float],
    window_days: int = 30,
    provider_stats: list[dict[str, Any]] | None = None,
) -> tuple[dict[int, float], dict[int, TierAdjustment]]:
    """One-shot helper for the orchestrator: fetch stats from Postgres,
    compute adjustments, return the dict the ``ResearchAgent`` expects
    as ``trust_weights``.

    Returns ``(weights_dict, adjustment_records)``. When Postgres is
    disabled or no stats exist, the base weights pass through
    unchanged and the adjustment records still show "no change" with
    zero calls so the operator UI has consistent rows.

    Pass ``provider_stats`` directly to skip the Postgres query — the
    operator-facing endpoint queries once and forwards both the raw
    stats and the derived adjustments to the response, avoiding a
    duplicate aggregation.
    """

    if provider_stats is None:
        if metrics_client is None:
            provider_stats = []
        else:
            try:
                from dark_factory.metrics.refinery_repository import (
                    RefineryMetricsRepository,
                )

                repo = RefineryMetricsRepository(metrics_client)
                provider_stats = repo.compute_provider_stats(
                    window_days=window_days,
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(
                    "refinery_provider_learning_query_failed", error=str(exc),
                )
                provider_stats = []

    adjustments = adjust_trust_weights(
        base_weights=base_weights, provider_stats=provider_stats,
    )
    weights = {
        tier: adj.adjusted_weight for tier, adj in adjustments.items()
    }
    return weights, adjustments


__all__ = [
    "TierAdjustment",
    "adjust_trust_weights",
    "load_adjusted_trust_weights",
]
