"""Tests for adaptive max_rounds per requirement archetype.

Three layers, mirroring test_role_weighting / test_judge_calibration:

1. ``adjust_archetype_rounds`` — bounded effective max_rounds math.
2. ``effective_max_rounds_for`` — identity-default lookup + key bucketing.
3. ``load_archetype_rounds`` — precomputed-stats path (skips Postgres).
"""

from __future__ import annotations

from dark_factory.api.refinery.archetype_rounds import (
    adjust_archetype_rounds,
    archetype_key_for,
    effective_max_rounds_for,
    load_archetype_rounds,
)


# ─────────────────────────────────────────────────────────────────────
# 1. adjust_archetype_rounds — bounded math
# ─────────────────────────────────────────────────────────────────────


def test_below_min_total_keeps_base():
    """An archetype with too few debates must NOT be re-tuned."""

    stats = [{
        "priority": "high", "source_mode": "run", "primary_tag": "auth",
        "total": 4, "converged": 3, "short_circuited": 1, "aborted": 0,
        "avg_rounds_converged": 2.0,
    }]
    out = adjust_archetype_rounds(archetype_stats=stats, base_max_rounds=3)
    key = archetype_key_for(priority="high", source_mode="run", tags=["auth"])
    assert out[key].effective_max_rounds == 3
    assert out[key].direction == "insufficient_signal"


def test_high_short_circuit_rate_bumps_two():
    """Archetype where most debates short-circuit needs more rounds."""

    stats = [{
        "priority": "critical", "source_mode": "documents",
        "primary_tag": "compliance",
        "total": 12, "converged": 4, "short_circuited": 8, "aborted": 0,
        "avg_rounds_converged": 3.0,
    }]
    out = adjust_archetype_rounds(archetype_stats=stats, base_max_rounds=3)
    key = archetype_key_for(
        priority="critical", source_mode="documents", tags=["compliance"],
    )
    assert out[key].effective_max_rounds == 5  # 3 + 2, below ceiling
    assert out[key].direction == "bump_two"
    assert out[key].reason == "high_short_circuit_rate"


def test_elevated_short_circuit_rate_bumps_one():
    stats = [{
        "priority": "high", "source_mode": "run", "primary_tag": "perf",
        "total": 10, "converged": 6, "short_circuited": 4, "aborted": 0,
        "avg_rounds_converged": 3.0,
    }]
    out = adjust_archetype_rounds(archetype_stats=stats, base_max_rounds=3)
    key = archetype_key_for(priority="high", source_mode="run", tags=["perf"])
    assert out[key].effective_max_rounds == 4
    assert out[key].direction == "bump_one"


def test_low_mean_rounds_reduces_by_one():
    """Archetype that converges fast gets a lower cap to save tokens."""

    stats = [{
        "priority": "low", "source_mode": "direct",
        "primary_tag": "internal-tooling",
        "total": 20, "converged": 18, "short_circuited": 1, "aborted": 1,
        # 1.5 < base*0.55 = 2.2 → reduce
        "avg_rounds_converged": 1.5,
    }]
    out = adjust_archetype_rounds(archetype_stats=stats, base_max_rounds=4)
    key = archetype_key_for(
        priority="low", source_mode="direct", tags=["internal-tooling"],
    )
    assert out[key].effective_max_rounds == 3  # 4 - 1
    assert out[key].direction == "reduce"


def test_reduce_respects_floor():
    """Reduce never goes below the hard floor of 2."""

    stats = [{
        "priority": "medium", "source_mode": "run", "primary_tag": "ops",
        "total": 30, "converged": 28, "short_circuited": 1, "aborted": 1,
        "avg_rounds_converged": 0.5,
    }]
    out = adjust_archetype_rounds(archetype_stats=stats, base_max_rounds=2)
    key = archetype_key_for(priority="medium", source_mode="run", tags=["ops"])
    assert out[key].effective_max_rounds == 2  # floor


def test_bump_respects_ceiling():
    """Bump never exceeds the hard ceiling of 6."""

    stats = [{
        "priority": "critical", "source_mode": "documents",
        "primary_tag": "compliance",
        "total": 25, "converged": 5, "short_circuited": 20, "aborted": 0,
        "avg_rounds_converged": 5.0,
    }]
    out = adjust_archetype_rounds(archetype_stats=stats, base_max_rounds=5)
    key = archetype_key_for(
        priority="critical", source_mode="documents", tags=["compliance"],
    )
    assert out[key].effective_max_rounds == 6  # ceiling


def test_within_tolerance_keeps_identity():
    """Mid-range stats with no clear signal → identity."""

    stats = [{
        "priority": "medium", "source_mode": "run", "primary_tag": "api",
        "total": 12, "converged": 10, "short_circuited": 2, "aborted": 0,
        "avg_rounds_converged": 2.5,  # 2.5 / 3 ≈ 0.83, above _REDUCE_RATIO
    }]
    out = adjust_archetype_rounds(archetype_stats=stats, base_max_rounds=3)
    key = archetype_key_for(priority="medium", source_mode="run", tags=["api"])
    assert out[key].effective_max_rounds == 3
    assert out[key].direction == "identity"


def test_bump_takes_priority_over_reduce_signal():
    """When both signals fire (rare), bump wins — a starved panel can't
    benefit from a tighter cap."""

    stats = [{
        "priority": "high", "source_mode": "run", "primary_tag": "auth",
        # Most short-circuit; the few that converge do so fast.
        "total": 20, "converged": 8, "short_circuited": 12, "aborted": 0,
        "avg_rounds_converged": 1.2,
    }]
    out = adjust_archetype_rounds(archetype_stats=stats, base_max_rounds=3)
    key = archetype_key_for(priority="high", source_mode="run", tags=["auth"])
    assert out[key].direction in {"bump_one", "bump_two"}
    assert out[key].effective_max_rounds > 3


def test_bad_row_is_skipped():
    """A malformed row doesn't crash the aggregator."""

    stats = [
        {"priority": "high", "source_mode": "run", "primary_tag": "x",
         "total": "not-an-int", "converged": 1, "short_circuited": 0},
        {"priority": "low", "source_mode": "run", "primary_tag": "y",
         "total": 12, "converged": 10, "short_circuited": 2, "aborted": 0,
         "avg_rounds_converged": 2.5},
    ]
    out = adjust_archetype_rounds(archetype_stats=stats, base_max_rounds=3)
    # Bad row dropped, good row present
    assert len(out) == 1
    assert next(iter(out.keys())) == archetype_key_for(
        priority="low", source_mode="run", tags=["y"],
    )


# ─────────────────────────────────────────────────────────────────────
# 2. effective_max_rounds_for — identity-default lookup
# ─────────────────────────────────────────────────────────────────────


def test_unknown_archetype_returns_base():
    """Lookup for an archetype with no learned entry returns the base."""

    stats = [{
        "priority": "high", "source_mode": "run", "primary_tag": "auth",
        "total": 12, "converged": 10, "short_circuited": 2, "aborted": 0,
        "avg_rounds_converged": 2.5,
    }]
    archetypes = adjust_archetype_rounds(archetype_stats=stats, base_max_rounds=3)
    eff = effective_max_rounds_for(
        priority="critical", source_mode="documents",
        tags=["something-else"], archetypes=archetypes, base_max_rounds=3,
    )
    assert eff == 3


def test_known_archetype_returns_tuned_value():
    stats = [{
        "priority": "low", "source_mode": "direct", "primary_tag": "ops",
        "total": 20, "converged": 18, "short_circuited": 1, "aborted": 1,
        "avg_rounds_converged": 1.0,
    }]
    archetypes = adjust_archetype_rounds(archetype_stats=stats, base_max_rounds=4)
    eff = effective_max_rounds_for(
        priority="low", source_mode="direct", tags=["ops", "internal"],
        archetypes=archetypes, base_max_rounds=4,
    )
    assert eff == 3  # reduced from 4


def test_archetype_key_normalization():
    """Keys are case-insensitive and trim whitespace; first non-empty
    tag is the primary tag."""

    a = archetype_key_for(priority="High", source_mode="RUN", tags=["Auth"])
    b = archetype_key_for(priority="high", source_mode="run", tags=["auth"])
    assert a == b

    # Empty tag list → "untagged"
    c = archetype_key_for(priority="high", source_mode="run", tags=None)
    assert c.endswith("|untagged")

    # First non-empty wins; whitespace-only ignored
    d = archetype_key_for(
        priority="high", source_mode="run", tags=["", "  ", "real-tag"],
    )
    assert d.endswith("|real-tag")


# ─────────────────────────────────────────────────────────────────────
# 3. load_archetype_rounds — precomputed-stats path skips Postgres
# ─────────────────────────────────────────────────────────────────────


def test_load_no_metrics_client_returns_empty():
    out = load_archetype_rounds(metrics_client=None, base_max_rounds=3)
    assert out == {}


def test_load_with_precomputed_stats_skips_postgres():
    out = load_archetype_rounds(
        metrics_client=None,
        base_max_rounds=3,
        archetype_stats=[{
            "priority": "critical", "source_mode": "documents",
            "primary_tag": "compliance",
            "total": 12, "converged": 4, "short_circuited": 8, "aborted": 0,
            "avg_rounds_converged": 3.0,
        }],
    )
    key = archetype_key_for(
        priority="critical", source_mode="documents", tags=["compliance"],
    )
    assert out[key].direction == "bump_two"
    assert out[key].effective_max_rounds == 5
