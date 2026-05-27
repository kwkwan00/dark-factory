"""Tests for the per-(role, dimension) credibility weighting loop.

Four layers, mirroring test_role_weighting.py with one extra:

1. ``adjust_role_dimension_weights`` — bounded multiplier math, split
   on the dimension axis.
2. ``dimension_weight_for`` / ``combined_weight_for`` — identity-default
   lookup, multiplicative composition with the flat per-role weight,
   product clamp.
3. ``load_role_dimension_weights`` — precomputed-stats path skips
   Postgres.
4. ``format_defend_prompt`` — renders the 2D table when both signals
   are stamped on the evidence bag, falls back to the 1D rendering
   when only the flat weights are present.
"""

from __future__ import annotations

import pytest

from dark_factory.api.refinery.contracts import (
    Critique,
    CritiqueDimension,
    Draft,
    DraftSpec,
    Severity,
)
from dark_factory.api.refinery.role_weighting import (
    RoleDimensionWeight,
    adjust_role_dimension_weights,
    combined_weight_for,
    dimension_weight_for,
    flatten_role_dim_weights,
    load_role_dimension_weights,
)
from dark_factory.api.refinery.roles.judge.prompt import format_defend_prompt


# ─────────────────────────────────────────────────────────────────────
# 1. adjust_role_dimension_weights — bounded math
# ─────────────────────────────────────────────────────────────────────


def test_below_min_total_yields_identity_per_cell():
    """Per-(role, dim) cells with too few rows in the window are NOT
    re-weighted — early signal isn't trustworthy. The dim threshold (6)
    is tighter than the flat one (8) because 5× shard makes signal
    sparser, but identity-default is the same."""

    stats = [{
        "role": "security", "dimension": "risk_coverage",
        "severity": "blocker",
        "accepted": 3, "rejected": 1, "deferred": 0, "total": 4,
    }]
    weights = adjust_role_dimension_weights(role_dim_stats=stats)
    entry = weights[("security", "risk_coverage")]
    assert entry.multiplier == pytest.approx(1.0)
    assert entry.reason == "below_min_total"


def test_high_acceptance_amplifies_to_dim_ceiling():
    """Roles with high acceptance on a specific dim get amplified — the
    dim ceiling (1.50) is wider than the flat ceiling (1.40) so a
    role's wheelhouse can lift further than the flat band allows."""

    stats = [{
        "role": "security", "dimension": "risk_coverage",
        "severity": "blocker",
        "accepted": 7, "rejected": 1, "deferred": 0, "total": 8,
    }]
    weights = adjust_role_dimension_weights(role_dim_stats=stats)
    entry = weights[("security", "risk_coverage")]
    # Default 1.0 + step 0.10, dim ceiling 1.50.
    assert entry.multiplier == pytest.approx(1.10)
    assert entry.reason == "high_acceptance"


def test_low_acceptance_attenuates_to_dim_floor():
    """A role's findings outside its wheelhouse get attenuated — but
    never below the dim floor (0.50). Identity floor never silences."""

    stats = [{
        "role": "cost", "dimension": "risk_coverage",
        "severity": "blocker",
        "accepted": 1, "rejected": 7, "deferred": 0, "total": 8,
    }]
    weights = adjust_role_dimension_weights(role_dim_stats=stats)
    entry = weights[("cost", "risk_coverage")]
    assert entry.multiplier == pytest.approx(0.90)
    assert entry.reason == "low_acceptance"
    assert entry.multiplier >= 0.50


def test_neutral_acceptance_stays_identity():
    """Mid-range acceptance keeps the multiplier at 1.0 — no signal
    in either direction."""

    stats = [{
        "role": "engineering", "dimension": "feasibility",
        "severity": "blocker",
        "accepted": 4, "rejected": 4, "deferred": 0, "total": 8,
    }]
    weights = adjust_role_dimension_weights(role_dim_stats=stats)
    entry = weights[("engineering", "feasibility")]
    assert entry.multiplier == pytest.approx(1.0)
    assert entry.reason == "neutral"


def test_severity_filter_drops_warnings():
    """Only blocker rows enter the math by default — accepting a
    WARNING is much cheaper than accepting a BLOCKER."""

    stats = [
        {"role": "security", "dimension": "risk_coverage",
         "severity": "warning",
         "accepted": 8, "rejected": 0, "deferred": 0, "total": 8},
        {"role": "security", "dimension": "risk_coverage",
         "severity": "blocker",
         "accepted": 1, "rejected": 7, "deferred": 0, "total": 8},
    ]
    weights = adjust_role_dimension_weights(
        role_dim_stats=stats, severity="blocker",
    )
    # Only the blocker row drives the cell.
    entry = weights[("security", "risk_coverage")]
    assert entry.reason == "low_acceptance"


def test_bad_row_is_skipped():
    """A malformed row doesn't crash the aggregator."""

    stats = [
        {"role": None, "dimension": "x", "severity": "blocker",
         "total": 10, "accepted": 5, "rejected": 5, "deferred": 0},
        {"role": "security", "dimension": "risk_coverage",
         "severity": "blocker",
         "accepted": 7, "rejected": 1, "deferred": 0, "total": 8},
    ]
    weights = adjust_role_dimension_weights(role_dim_stats=stats)
    assert ("security", "risk_coverage") in weights
    assert ("None", "x") not in weights


# ─────────────────────────────────────────────────────────────────────
# 2. lookup helpers — identity default + multiplicative composition
# ─────────────────────────────────────────────────────────────────────


def test_dimension_weight_for_unknown_cell_is_identity():
    """Unknown ``(role, dimension)`` returns 1.0 — never silently
    demote a role on a dimension we have no signal for."""

    assert dimension_weight_for("security", "feasibility", {}) == 1.0
    assert dimension_weight_for(
        "anyone", "anything",
        {("security", "risk_coverage"): RoleDimensionWeight(
            role="security", dimension="risk_coverage",
            multiplier=1.5, accepted=7, rejected=1, deferred=0,
            total=8, acceptance_rate=0.875, reason="high_acceptance",
        )},
    ) == 1.0


def test_dimension_weight_for_accepts_flat_evidence_shape():
    """Accept the JSON-friendly ``"role:dim" → float`` dict shape so
    the evidence bag can transit values without typed objects."""

    flat = {"security:risk_coverage": 1.20, "cost:feasibility": 0.85}
    assert dimension_weight_for("security", "risk_coverage", flat) == 1.20
    assert dimension_weight_for("cost", "feasibility", flat) == 0.85
    # Unknown still returns identity
    assert dimension_weight_for("cost", "risk_coverage", flat) == 1.0


def test_combined_weight_clamps_to_product_band():
    """The product of role × dim is clamped to ``[0.50, 1.60]`` so
    composing the bounds of two factors can't escape into ranges
    neither was designed for."""

    role_w = {"security": 1.40}            # at flat ceiling
    dim_w = {"security:risk_coverage": 1.50}  # at dim ceiling
    # Mathematical product = 2.10; clamped to 1.60.
    combined = combined_weight_for(
        "security", "risk_coverage",
        role_weights=role_w, role_dim_weights=dim_w,
    )
    assert combined == pytest.approx(1.60)

    # Worst case: both at floor → 0.60 × 0.50 = 0.30, clamped to 0.50.
    role_w_low = {"cost": 0.60}
    dim_w_low = {"cost:risk_coverage": 0.50}
    combined_low = combined_weight_for(
        "cost", "risk_coverage",
        role_weights=role_w_low, role_dim_weights=dim_w_low,
    )
    assert combined_low == pytest.approx(0.50)


def test_combined_weight_inside_band_passes_through():
    """When the natural product falls inside the band, the clamp is a
    no-op — the LLM sees the actual composed signal."""

    combined = combined_weight_for(
        "security", "risk_coverage",
        role_weights={"security": 1.10},
        role_dim_weights={"security:risk_coverage": 1.20},
    )
    assert combined == pytest.approx(1.32)


def test_combined_weight_identity_when_both_missing():
    assert combined_weight_for(
        "anyone", "anything",
        role_weights={}, role_dim_weights={},
    ) == 1.0
    assert combined_weight_for(
        "anyone", "anything",
        role_weights=None, role_dim_weights=None,
    ) == 1.0


def test_flatten_drops_identity_cells():
    """The wire-format dict only carries non-default cells — identity
    is implicit at the lookup site, and keeping the payload small keeps
    the synthesis prompt's calibration block readable."""

    weights = adjust_role_dimension_weights(role_dim_stats=[
        # Non-identity cell
        {"role": "security", "dimension": "risk_coverage",
         "severity": "blocker",
         "accepted": 7, "rejected": 1, "deferred": 0, "total": 8},
        # Identity cell (neutral)
        {"role": "engineering", "dimension": "feasibility",
         "severity": "blocker",
         "accepted": 4, "rejected": 4, "deferred": 0, "total": 8},
    ])
    flat = flatten_role_dim_weights(weights)
    assert "security:risk_coverage" in flat
    assert "engineering:feasibility" not in flat


# ─────────────────────────────────────────────────────────────────────
# 3. load_role_dimension_weights — precomputed-stats path
# ─────────────────────────────────────────────────────────────────────


def test_load_no_metrics_client_returns_empty():
    out = load_role_dimension_weights(metrics_client=None)
    assert out == {}


def test_load_with_precomputed_stats_skips_postgres():
    out = load_role_dimension_weights(
        metrics_client=None,
        role_dim_stats=[{
            "role": "security", "dimension": "risk_coverage",
            "severity": "blocker",
            "accepted": 7, "rejected": 1, "deferred": 0, "total": 8,
        }],
    )
    assert ("security", "risk_coverage") in out


# ─────────────────────────────────────────────────────────────────────
# 4. format_defend_prompt — 2D rendering vs 1D fallback
# ─────────────────────────────────────────────────────────────────────


def _draft() -> Draft:
    return Draft(
        requirement_id="r-1", title="t", description="d",
        priority="medium", tags=[],
        suggested_specs=[
            DraftSpec(title="s", capability="auth", description="x",
                      acceptance_criteria=["a", "b", "c"]),
        ],
        relationships=[], produced_by="product", iteration=0,
    )


def _critiques() -> list[Critique]:
    return [
        Critique(
            author_role="security", severity=Severity.BLOCKER,
            dimension=CritiqueDimension.RISK_COVERAGE,
            finding="finding text long enough to satisfy validators",
            proposed_fix="fix text",
        ),
    ]


def test_prompt_renders_2d_table_when_both_signals_stamped():
    """When per-(role, dim) weights are stamped alongside flat per-
    role weights, the prompt must show the COMBINED multiplier per
    cell — the LLM doesn't have to compose factors itself."""

    prompt = format_defend_prompt(
        _draft(), _critiques(), iteration=1,
        role_weights={"security": 1.10},
        role_dim_weights={"security:risk_coverage": 1.20},
    )
    assert "Role × dimension calibration" in prompt
    # Combined = 1.10 × 1.20 = 1.32
    assert "×1.32" in prompt
    assert "security on **risk_coverage**" in prompt
    # And the legacy 1D block must NOT also appear
    assert "Role calibration (historical acceptance signal)" not in prompt


def test_prompt_falls_back_to_1d_when_only_flat_weights():
    """No per-dim signal yet → use the legacy 1D block to preserve
    early-life behaviour and existing tests."""

    prompt = format_defend_prompt(
        _draft(), _critiques(), iteration=1,
        role_weights={"security": 1.10},
        role_dim_weights=None,
    )
    assert "Role calibration (historical acceptance signal)" in prompt
    assert "Role × dimension calibration" not in prompt
    assert "security: ×1.10" in prompt


def test_prompt_omits_block_entirely_when_no_signals():
    """Both signals empty → no calibration block (matches early-life
    deployment behaviour)."""

    prompt = format_defend_prompt(
        _draft(), _critiques(), iteration=1,
        role_weights=None, role_dim_weights=None,
    )
    assert "Role calibration" not in prompt
    assert "Role × dimension calibration" not in prompt


def test_prompt_2d_table_drops_identity_cells():
    """A cell where the combined multiplier is ≈1.0 isn't worth a row
    — the table only renders cells with a real signal."""

    prompt = format_defend_prompt(
        _draft(), _critiques(), iteration=1,
        # security on risk_coverage: 1.10 × 1.20 = 1.32 → keep.
        # security on feasibility: implicit identity 1.10 × 1.0 = 1.10 → keep.
        # cost on feasibility: 1.0 × 1.0 = 1.0 → drop.
        role_weights={"security": 1.10, "cost": 1.0},
        role_dim_weights={
            "security:risk_coverage": 1.20,
            "security:feasibility": 1.0,
        },
    )
    assert "security on **risk_coverage**" in prompt
    assert "cost on **feasibility**" not in prompt
