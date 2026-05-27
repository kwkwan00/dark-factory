"""Tests for Judge confidence calibration.

Three layers:

1. ``adjust_judge_threshold`` — bounded multiplier math.
2. ``load_judge_calibration`` — precomputed-stats path.
3. ``CombinedJudgePipeline.score`` — reads the multiplier off the
   evidence bag and scales the effective threshold.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dark_factory.api.refinery.contracts import (
    Draft,
    DraftSpec,
    RoleContext,
)
from dark_factory.api.refinery.judge_calibration import (
    JudgeCalibration,
    adjust_judge_threshold,
    load_judge_calibration,
)


# ─────────────────────────────────────────────────────────────────────
# 1. adjust_judge_threshold — bounded multiplier math
# ─────────────────────────────────────────────────────────────────────


def test_no_stats_returns_identity():
    cal = adjust_judge_threshold(None)
    assert cal.multiplier == pytest.approx(1.0)
    assert cal.direction == "insufficient_signal"


def test_below_min_total_returns_identity():
    """Fewer than _MIN_TOTAL_FOR_LEARNING decisions → identity."""

    cal = adjust_judge_threshold({
        "high_applied": 2, "high_dismissed": 1,
        "low_applied": 0, "low_dismissed": 1,
    })
    assert cal.multiplier == pytest.approx(1.0)
    assert cal.direction == "insufficient_signal"
    assert cal.reason == "below_min_total"


def test_calibrated_within_tolerance_keeps_identity():
    """Miscalibration rate below trigger → no nudge even with enough
    total decisions. The Judge is performing within the tolerance band."""

    cal = adjust_judge_threshold({
        "high_applied": 8, "high_dismissed": 1,
        "low_applied": 1, "low_dismissed": 5,
    })
    # Miscalibration = (1+1)/15 ≈ 0.13, below 0.30 trigger.
    assert cal.multiplier == pytest.approx(1.0)
    assert cal.direction == "calibrated"
    assert cal.miscalibration_rate < 0.30


def test_overconfident_judge_raises_threshold():
    """High-score → dismissed dominates → multiplier > 1.0 (harder to
    pass), bounded at 1.15."""

    cal = adjust_judge_threshold({
        "high_applied": 2, "high_dismissed": 8,
        "low_applied": 1, "low_dismissed": 5,
    })
    # Miscalibration = (8+1)/16 ≈ 0.56 > trigger; high_dismissed > low_applied.
    assert cal.multiplier == pytest.approx(1.05)
    assert cal.direction == "overconfident"
    assert cal.multiplier <= 1.15


def test_underconfident_judge_lowers_threshold():
    """Low-score → applied dominates → multiplier < 1.0 (easier to
    pass), bounded at 0.85."""

    cal = adjust_judge_threshold({
        "high_applied": 4, "high_dismissed": 1,
        "low_applied": 8, "low_dismissed": 1,
    })
    # Miscalibration = (1+8)/14 ≈ 0.64 > trigger; low_applied > high_dismissed.
    assert cal.multiplier == pytest.approx(0.95)
    assert cal.direction == "underconfident"
    assert cal.multiplier >= 0.85


def test_balanced_error_arms_keep_identity():
    """Both error arms equal → no clear direction → no nudge.

    The miscalibration rate may be high, but there's no signal about
    which way to move the threshold."""

    cal = adjust_judge_threshold({
        "high_applied": 2, "high_dismissed": 5,
        "low_applied": 5, "low_dismissed": 2,
    })
    assert cal.multiplier == pytest.approx(1.0)
    assert cal.reason == "balanced_error_arms"


# ─────────────────────────────────────────────────────────────────────
# 2. load_judge_calibration — precomputed stats path
# ─────────────────────────────────────────────────────────────────────


def test_load_calibration_no_metrics_client_returns_identity():
    cal = load_judge_calibration(metrics_client=None)
    assert cal.multiplier == pytest.approx(1.0)
    assert cal.direction == "insufficient_signal"


def test_load_calibration_with_precomputed_stats_skips_postgres():
    """Passing precomputed stats lets dashboards reuse one query."""

    cal = load_judge_calibration(
        metrics_client=None,
        stats={
            "high_applied": 2, "high_dismissed": 8,
            "low_applied": 1, "low_dismissed": 5,
        },
    )
    assert cal.direction == "overconfident"


# ─────────────────────────────────────────────────────────────────────
# 3. CombinedJudgePipeline.score reads multiplier off evidence bag
# ─────────────────────────────────────────────────────────────────────


def _draft() -> Draft:
    return Draft(
        requirement_id="r-1",
        title="OAuth2 authorization-code flow",
        description=(
            "Given a valid client when authorize is called then a redirect "
            "occurs. Response time must be under 300ms at 100 req/s."
        ),
        priority="high",
        tags=["auth", "security"],
        suggested_specs=[
            DraftSpec(
                title="OAuth2 flow", capability="auth", description="...",
                acceptance_criteria=[
                    "given client when authorize then redirect",
                    "given missing client when authorize then 400",
                    "given expired token when refresh then new token issued",
                ],
            ),
        ],
        relationships=[], produced_by="product", iteration=0,
    )


def test_pipeline_uses_default_threshold_without_multiplier():
    """No multiplier in evidence → effective threshold == base; the
    EvaluationScore's overall_threshold field reflects the base."""

    from dark_factory.api.refinery.judge.composition import (
        CombinedJudgePipeline,
    )
    from dark_factory.api.refinery.judge.fallback_judge import FallbackJudge
    from dark_factory.api.refinery.judge.rules_judge import RulesJudge

    pipeline = CombinedJudgePipeline(
        rules_judge=RulesJudge(),
        semantic_judge=FallbackJudge(overall_threshold=0.8),
        fallback_judge=FallbackJudge(overall_threshold=0.8),
        overall_threshold=0.8,
    )
    score = pipeline.score(
        _draft(),
        RoleContext(role="judge", requirement_id="r-1", evidence={}),
    )
    assert score.overall_threshold == pytest.approx(0.8)


def test_pipeline_scales_threshold_when_multiplier_in_evidence():
    """An overconfident-Judge multiplier (1.05) must yield a higher
    effective threshold (0.84) and that's what the EvaluationScore
    reports — not the base 0.8."""

    from dark_factory.api.refinery.judge.composition import (
        CombinedJudgePipeline,
    )
    from dark_factory.api.refinery.judge.fallback_judge import FallbackJudge
    from dark_factory.api.refinery.judge.rules_judge import RulesJudge

    pipeline = CombinedJudgePipeline(
        rules_judge=RulesJudge(),
        semantic_judge=FallbackJudge(overall_threshold=0.8),
        fallback_judge=FallbackJudge(overall_threshold=0.8),
        overall_threshold=0.8,
    )
    score = pipeline.score(
        _draft(),
        RoleContext(
            role="judge", requirement_id="r-1",
            evidence={"judge_threshold_multiplier": 1.05},
        ),
    )
    assert score.overall_threshold == pytest.approx(0.84)


def test_pipeline_uses_underconfident_multiplier_to_lower_threshold():
    """Multiplier < 1.0 → easier to pass; the EvaluationScore reflects
    the lowered effective threshold."""

    from dark_factory.api.refinery.judge.composition import (
        CombinedJudgePipeline,
    )
    from dark_factory.api.refinery.judge.fallback_judge import FallbackJudge
    from dark_factory.api.refinery.judge.rules_judge import RulesJudge

    pipeline = CombinedJudgePipeline(
        rules_judge=RulesJudge(),
        semantic_judge=FallbackJudge(overall_threshold=0.8),
        fallback_judge=FallbackJudge(overall_threshold=0.8),
        overall_threshold=0.8,
    )
    score = pipeline.score(
        _draft(),
        RoleContext(
            role="judge", requirement_id="r-1",
            evidence={"judge_threshold_multiplier": 0.95},
        ),
    )
    assert score.overall_threshold == pytest.approx(0.76)
