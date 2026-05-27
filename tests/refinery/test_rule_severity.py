"""Tests for self-tuning rule severity (blocker → warning demotion).

Three layers:

1. ``adjust_rule_severities`` — bounded demote-only math.
2. ``apply_severity_overrides`` — RuleResult remap (no in-place mutation).
3. ``CombinedJudgePipeline.score`` — reads ``rule_severity_overrides``
   from evidence and applies it before Phase B / Phase C.
"""

from __future__ import annotations

import pytest

from dark_factory.api.refinery.contracts import (
    CritiqueDimension,
    Draft,
    DraftSpec,
    RoleContext,
    RuleViolation,
    Severity,
)
from dark_factory.api.refinery.judge.rules_judge import RuleResult
from dark_factory.api.refinery.rule_severity import (
    RuleSeverityMap,
    adjust_rule_severities,
    apply_severity_overrides,
    load_rule_severities,
)


# ─────────────────────────────────────────────────────────────────────
# 1. adjust_rule_severities — bounded demote-only math
# ─────────────────────────────────────────────────────────────────────


def test_below_min_total_keeps_blocker():
    """A rule with too few decided blockers is not demoted."""

    out = adjust_rule_severities([{
        "rule_id": "rule_description_specificity",
        "blockers": 5, "decided": 3, "overrides": 3, "dismissals": 0,
    }])
    entry = out.overrides["rule_description_specificity"]
    assert entry.effective_severity == Severity.BLOCKER
    assert entry.direction == "insufficient_signal"
    assert entry.reason == "below_min_total"


def test_high_override_rate_demotes_to_warning():
    """A blocker that operators apply ≥50% of the time gets demoted."""

    out = adjust_rule_severities([{
        "rule_id": "rule_acceptance_criteria_min_count",
        "blockers": 20, "decided": 12, "overrides": 8, "dismissals": 4,
    }])
    entry = out.overrides["rule_acceptance_criteria_min_count"]
    assert entry.effective_severity == Severity.WARNING
    assert entry.direction == "demoted"
    assert entry.override_rate == pytest.approx(8 / 12)


def test_low_override_rate_keeps_blocker():
    """When operators dismiss most blockers the rule was right — keep it."""

    out = adjust_rule_severities([{
        "rule_id": "rule_no_self_reference",
        "blockers": 15, "decided": 10, "overrides": 1, "dismissals": 9,
    }])
    entry = out.overrides["rule_no_self_reference"]
    assert entry.effective_severity == Severity.BLOCKER
    assert entry.direction == "identity"


def test_unknown_rule_lookup_returns_base():
    """A rule not in the override map keeps its caller-supplied default."""

    out = adjust_rule_severities([])
    assert out.severity_for("nonexistent_rule", Severity.BLOCKER) == Severity.BLOCKER
    assert out.severity_for("nonexistent_rule", Severity.WARNING) == Severity.WARNING


def test_to_flat_only_includes_demotions():
    """The wire-format dict carries only demoted entries — identity is
    implicit at the lookup site."""

    out = adjust_rule_severities([
        {"rule_id": "demoted_rule", "blockers": 20, "decided": 12,
         "overrides": 9, "dismissals": 3},
        {"rule_id": "kept_rule", "blockers": 20, "decided": 12,
         "overrides": 1, "dismissals": 11},
        {"rule_id": "insufficient_rule", "blockers": 4, "decided": 4,
         "overrides": 4, "dismissals": 0},
    ])
    flat = out.to_flat()
    assert flat == {"demoted_rule": "warning"}


def test_bad_row_is_skipped():
    out = adjust_rule_severities([
        {"rule_id": "good_rule", "blockers": 12, "decided": 12,
         "overrides": 8, "dismissals": 4},
        {"rule_id": "bad_rule", "blockers": "not-an-int",
         "decided": 8, "overrides": 4},
    ])
    assert "good_rule" in out.overrides
    assert "bad_rule" not in out.overrides


# ─────────────────────────────────────────────────────────────────────
# 2. apply_severity_overrides — RuleResult remap
# ─────────────────────────────────────────────────────────────────────


def _viol(rule_id: str, severity: Severity = Severity.BLOCKER) -> RuleViolation:
    return RuleViolation(
        rule_id=rule_id, severity=severity,
        dimension=CritiqueDimension.CLARITY,
        finding=f"finding for {rule_id}",
        suggested_fix="fix it",
    )


def test_apply_demotes_named_rule_to_warning():
    rr = RuleResult(violations=[
        _viol("rule_a"),
        _viol("rule_b"),
    ])
    flat = {"rule_a": "warning"}
    out = apply_severity_overrides(rr, flat)
    severities = {v.rule_id: v.severity for v in out.violations}
    assert severities["rule_a"] == Severity.WARNING
    assert severities["rule_b"] == Severity.BLOCKER


def test_apply_warning_is_never_promoted():
    """The loop is demote-only; an already-warning row passes through."""

    rr = RuleResult(violations=[_viol("rule_a", Severity.WARNING)])
    out = apply_severity_overrides(rr, {"rule_a": "blocker"})
    assert out.violations[0].severity == Severity.WARNING


def test_apply_no_overrides_returns_input_unchanged():
    rr = RuleResult(violations=[_viol("rule_a")])
    out = apply_severity_overrides(rr, None)
    assert out is rr  # short-circuit
    out2 = apply_severity_overrides(rr, {})
    assert out2 is rr


def test_apply_does_not_mutate_input():
    rr = RuleResult(violations=[_viol("rule_a")])
    apply_severity_overrides(rr, {"rule_a": "warning"})
    # Original should still hold a blocker.
    assert rr.violations[0].severity == Severity.BLOCKER


def test_apply_preserves_dimension_and_finding():
    rr = RuleResult(violations=[_viol("rule_a")])
    out = apply_severity_overrides(rr, {"rule_a": "warning"})
    v = out.violations[0]
    assert v.dimension == CritiqueDimension.CLARITY
    assert v.finding == "finding for rule_a"
    assert v.suggested_fix == "fix it"


def test_apply_accepts_typed_severity_map():
    """The composition path may pass the typed map directly when run
    in-process; the flat dict is for the wire format."""

    severities = adjust_rule_severities([{
        "rule_id": "rule_a", "blockers": 20, "decided": 12,
        "overrides": 9, "dismissals": 3,
    }])
    rr = RuleResult(violations=[_viol("rule_a")])
    out = apply_severity_overrides(rr, severities)
    assert out.violations[0].severity == Severity.WARNING


# ─────────────────────────────────────────────────────────────────────
# 3. load_rule_severities — precomputed-stats path skips Postgres
# ─────────────────────────────────────────────────────────────────────


def test_load_no_metrics_client_returns_empty():
    out = load_rule_severities(metrics_client=None)
    assert isinstance(out, RuleSeverityMap)
    assert out.overrides == {}


def test_load_with_precomputed_stats_skips_postgres():
    out = load_rule_severities(
        metrics_client=None,
        rule_stats=[{
            "rule_id": "demoted_rule", "blockers": 20, "decided": 12,
            "overrides": 9, "dismissals": 3,
        }],
    )
    assert out.has_demotions()
    assert out.overrides["demoted_rule"].effective_severity == Severity.WARNING


# ─────────────────────────────────────────────────────────────────────
# 4. CombinedJudgePipeline reads overrides off evidence bag
# ─────────────────────────────────────────────────────────────────────


def _draft_with_blocker_trigger() -> Draft:
    """A draft that fires rule_acceptance_criteria_min_count (a real
    rule in the catalog — needs ≥ 3 acceptance_criteria per spec)."""

    return Draft(
        requirement_id="r-1",
        title="OAuth2 flow",
        description=(
            "Given a valid client when authorize is called then a redirect "
            "occurs. Response time must be under 300ms at 100 req/s."
        ),
        priority="high",
        tags=["auth", "security"],
        suggested_specs=[
            DraftSpec(
                title="OAuth2 spec", capability="auth", description="...",
                # only 1 criterion — triggers rule_acceptance_criteria_min_count
                acceptance_criteria=["given x when y then z"],
            ),
        ],
        relationships=[], produced_by="product", iteration=0,
    )


def test_pipeline_demotes_rule_when_evidence_carries_override():
    """When the evidence bag carries an override demoting a rule from
    blocker to warning, the pipeline's EvaluationScore must NOT carry
    the rule as a blocker, and Phase C clamps must apply the warning
    delta instead of the blocker cap.
    """

    from dark_factory.api.refinery.judge.composition import CombinedJudgePipeline
    from dark_factory.api.refinery.judge.fallback_judge import FallbackJudge
    from dark_factory.api.refinery.judge.rules_judge import RulesJudge

    pipeline = CombinedJudgePipeline(
        rules_judge=RulesJudge(),
        semantic_judge=FallbackJudge(overall_threshold=0.8),
        fallback_judge=FallbackJudge(overall_threshold=0.8),
        overall_threshold=0.8,
    )
    # Without override: blocker fires
    no_override = pipeline.score(
        _draft_with_blocker_trigger(),
        RoleContext(role="judge", requirement_id="r-1", evidence={}),
    )
    blockers_baseline = [v.rule_id for v in no_override.rule_violations]

    # With override on the same rule: it shows up as warning, not blocker
    overrides = {rid: "warning" for rid in blockers_baseline} or \
        {"rule_acceptance_criteria_min_count": "warning"}
    with_override = pipeline.score(
        _draft_with_blocker_trigger(),
        RoleContext(
            role="judge", requirement_id="r-1",
            evidence={"rule_severity_overrides": overrides},
        ),
    )
    # Demoted rules are absent from blockers and present in warnings.
    demoted_blockers = [v.rule_id for v in with_override.rule_violations]
    demoted_warnings = [v.rule_id for v in with_override.rule_warnings]
    for rid in overrides:
        assert rid not in demoted_blockers
        assert rid in demoted_warnings


def test_pipeline_no_op_when_evidence_lacks_overrides():
    """Identity behaviour when no override is on the bag — Phase A
    output passes through to Phase B/C unchanged."""

    from dark_factory.api.refinery.judge.composition import CombinedJudgePipeline
    from dark_factory.api.refinery.judge.fallback_judge import FallbackJudge
    from dark_factory.api.refinery.judge.rules_judge import RulesJudge

    pipeline = CombinedJudgePipeline(
        rules_judge=RulesJudge(),
        semantic_judge=FallbackJudge(overall_threshold=0.8),
        fallback_judge=FallbackJudge(overall_threshold=0.8),
        overall_threshold=0.8,
    )
    score = pipeline.score(
        _draft_with_blocker_trigger(),
        RoleContext(role="judge", requirement_id="r-1", evidence={}),
    )
    # The blocker still gates convergence
    assert score.passed is False
    assert any(v.severity == Severity.BLOCKER for v in score.rule_violations)
