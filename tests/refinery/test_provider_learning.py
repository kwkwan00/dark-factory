"""Tests for the T3/T5 provider learning loop."""

from __future__ import annotations

import pytest

from dark_factory.api.refinery.research.learning import (
    TierAdjustment,
    adjust_trust_weights,
)


def _stat(tier: int, provider: str, calls: int, propagated: int):
    return {
        "tier": tier,
        "provider": provider,
        "calls": calls,
        "propagated": propagated,
        "rate": (propagated / calls) if calls else 0.0,
        "mean_confidence": None,
    }


def test_no_signal_means_no_change():
    """Below the min-calls threshold, weights pass through unchanged."""

    base = {3: 0.80, 5: 0.30}
    out = adjust_trust_weights(
        base_weights=base,
        provider_stats=[_stat(3, "official", calls=2, propagated=2)],
    )
    assert out[3].adjusted_weight == 0.80
    assert "insufficient signal" in out[3].reason
    assert out[5].adjusted_weight == 0.30


def test_low_contribution_rate_demotes():
    """Sustained low contribution rate softly demotes the tier."""

    base = {5: 0.30}
    # 30 calls, only 1 propagated → rate ~3% (well below 10% floor).
    out = adjust_trust_weights(
        base_weights=base,
        provider_stats=[_stat(5, "web", calls=30, propagated=1)],
    )
    assert out[5].adjusted_weight < 0.30
    assert out[5].adjusted_weight >= 0.10  # bounded by per-tier floor
    assert "soft demotion" in out[5].reason


def test_high_contribution_rate_boosts():
    """Sustained high contribution rate softly boosts the tier."""

    base = {3: 0.80}
    # 20 calls, 12 propagated → rate 60% (well above 50% ceiling).
    out = adjust_trust_weights(
        base_weights=base,
        provider_stats=[_stat(3, "official", calls=20, propagated=12)],
    )
    assert out[3].adjusted_weight > 0.80
    assert out[3].adjusted_weight <= 0.90  # bounded by per-tier ceiling
    assert "soft boost" in out[3].reason


def test_per_tier_bounds_clamp_runaway_drift():
    """Per-tier (min, max) bounds clamp runaway adjustments so a
    learning loop can't push T5 above T0."""

    base = {5: 0.50}  # already at the T5 ceiling
    out = adjust_trust_weights(
        base_weights=base,
        provider_stats=[_stat(5, "web", calls=100, propagated=99)],
    )
    # 99% propagation rate would otherwise boost; bound holds.
    assert out[5].adjusted_weight <= 0.50


def test_aggregates_across_providers_per_tier():
    """Multiple providers feeding the same tier roll up into one
    contribution rate."""

    base = {3: 0.80}
    out = adjust_trust_weights(
        base_weights=base,
        provider_stats=[
            _stat(3, "official_aws", calls=10, propagated=2),
            _stat(3, "official_gcp", calls=10, propagated=1),
            _stat(3, "official_anthropic", calls=10, propagated=0),
        ],
    )
    # 30 total calls, 3 propagated → 10% — at the floor; still demotes
    # because the rate is < floor (strict-less). Acceptable either way;
    # we just need to confirm aggregation happened.
    assert out[3].calls == 30
    assert out[3].propagated == 3
    assert out[3].contribution_rate == pytest.approx(0.10)


def test_tier_adjustment_record_shape():
    out = adjust_trust_weights(
        base_weights={0: 1.0},
        provider_stats=[_stat(0, "structured", calls=20, propagated=15)],
    )
    rec = out[0]
    assert isinstance(rec, TierAdjustment)
    assert rec.tier == 0
    assert rec.calls == 20
    assert rec.propagated == 15
