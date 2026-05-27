"""Self-tuning rule severity — bounded blocker → warning demotion.

Mirrors the role-weighting and judge-calibration loops, but the signal
is **operator override behaviour on rule blockers**:

- A rule blocker fires → debate fails to converge or scores low →
  operator nevertheless **applies** the requirement → that's an
  *override*. The rule blocked something the operator considered fine
  to ship.
- A rule blocker fires → operator dismisses the requirement → the rule
  was right.

A high override rate signals a rule that's too strict for the deployment
context. Rather than asking operators to disable the rule entirely (an
all-or-nothing choice), the loop **demotes** the rule from blocker to
warning at runtime — its findings still surface in the trace and feed
the warning-penalty in Phase C, but they no longer gate convergence.

Hysteresis: only demote when the override rate is **clearly** high
(``>= _DEMOTE_TRIGGER``) and only when there's enough signal
(``>= _MIN_TOTAL_FOR_LEARNING`` decided blockers). The deadband stops
the severity from flapping when a rule's override rate hovers around the
trigger.

Identity-default everywhere. Promotion (warning → blocker) is not in
scope: warnings don't gate convergence, so warnings don't generate the
override-vs-dismiss signal we'd need to do this safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from dark_factory.api.refinery.contracts import (
    RuleViolation,
    Severity,
)
from dark_factory.api.refinery.judge.rules_judge import RuleResult

log = structlog.get_logger()


_MIN_TOTAL_FOR_LEARNING = 8
"""Below this many decided-on blocker firings, ignore the rule — too
little operator follow-through to trust the rate."""

_DEMOTE_TRIGGER = 0.50
"""Override rate at or above which a rule is demoted from blocker to
warning. Half the operators applying despite a blocker is the bar at
which the rule is doing more harm than good."""


@dataclass(frozen=True)
class RuleSeverityOverride:
    """Effective severity for one rule + audit trail."""

    rule_id: str
    effective_severity: Severity
    base_severity: Severity
    blockers: int
    decided: int
    overrides: int
    dismissals: int
    override_rate: float
    direction: str
    """``demoted`` | ``identity`` | ``insufficient_signal``"""
    reason: str


@dataclass
class RuleSeverityMap:
    """Per-rule severity overrides + the audit trail."""

    overrides: dict[str, RuleSeverityOverride] = field(default_factory=dict)

    def severity_for(
        self, rule_id: str, base_severity: Severity,
    ) -> Severity:
        """Identity-default lookup."""

        entry = self.overrides.get(rule_id)
        return entry.effective_severity if entry is not None else base_severity

    def has_demotions(self) -> bool:
        return any(
            o.direction == "demoted" for o in self.overrides.values()
        )

    def to_flat(self) -> dict[str, str]:
        """Flat ``{rule_id: severity_value}`` for the evidence bag.

        Only **demoted** entries are included — identity entries are
        implicit at the lookup site. Keeps the evidence_bag payload
        small and the wire format easy to inspect."""

        return {
            rule_id: o.effective_severity.value
            for rule_id, o in self.overrides.items()
            if o.direction == "demoted"
        }


def adjust_rule_severities(
    rule_stats: list[dict[str, Any]] | None,
) -> RuleSeverityMap:
    """Convert raw override stats into per-rule severity decisions.

    Rules absent from the input get no entry in the result; callers that
    want identity behaviour for missing rules use
    :meth:`RuleSeverityMap.severity_for`.
    """

    out = RuleSeverityMap()
    for row in rule_stats or []:
        rule_id = row.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            continue
        try:
            blockers = int(row.get("blockers") or 0)
            decided = int(row.get("decided") or 0)
            overrides = int(row.get("overrides") or 0)
            dismissals = int(row.get("dismissals") or 0)
        except (TypeError, ValueError):
            log.warning("refinery_rule_severity_bad_row", row=row)
            continue
        rate = (overrides / decided) if decided else 0.0

        if decided < _MIN_TOTAL_FOR_LEARNING:
            severity = Severity.BLOCKER
            direction = "insufficient_signal"
            reason = "below_min_total"
        elif rate >= _DEMOTE_TRIGGER:
            severity = Severity.WARNING
            direction = "demoted"
            reason = "override_rate_at_or_above_trigger"
        else:
            severity = Severity.BLOCKER
            direction = "identity"
            reason = "override_rate_below_trigger"

        out.overrides[rule_id] = RuleSeverityOverride(
            rule_id=rule_id,
            effective_severity=severity,
            base_severity=Severity.BLOCKER,
            blockers=blockers,
            decided=decided,
            overrides=overrides,
            dismissals=dismissals,
            override_rate=rate,
            direction=direction,
            reason=reason,
        )
    return out


def apply_severity_overrides(
    rule_result: RuleResult,
    severities: RuleSeverityMap | dict[str, str] | None,
) -> RuleResult:
    """Return a new ``RuleResult`` with violations remapped to the
    operator-tuned severity. The original is left untouched so the
    pre-override view stays available for the trace.

    ``severities`` accepts either the typed ``RuleSeverityMap`` (used by
    the loader path) or the flat ``{rule_id: severity_value}`` dict that
    travels on the evidence bag.
    """

    if not severities:
        return rule_result

    if isinstance(severities, RuleSeverityMap):
        flat = severities.to_flat()
    else:
        flat = dict(severities)

    if not flat:
        return rule_result

    new_violations: list[RuleViolation] = []
    for v in rule_result.violations:
        target = flat.get(v.rule_id)
        if target is None or v.severity == Severity.WARNING:
            # Warnings are never promoted by this loop; only blocker→warning
            # demotions take effect here.
            new_violations.append(v)
            continue
        try:
            new_severity = Severity(target)
        except ValueError:
            new_violations.append(v)
            continue
        if new_severity == v.severity:
            new_violations.append(v)
            continue
        # Preserve the rule's dimension and finding; only the severity
        # changes. Pydantic .model_copy keeps the rest of the row stable.
        try:
            new_violations.append(
                v.model_copy(update={"severity": new_severity}),
            )
        except Exception:
            new_violations.append(v)

    return RuleResult(
        violations=new_violations,
        rules_run=list(rule_result.rules_run),
        rules_skipped=list(rule_result.rules_skipped),
        engine_failed=rule_result.engine_failed,
    )


def load_rule_severities(
    *,
    metrics_client: Any | None,
    window_days: int = 30,
    rule_stats: list[dict[str, Any]] | None = None,
) -> RuleSeverityMap:
    """Top-level entry — fetch stats from Postgres (or take precomputed
    ones for endpoints that already have them) and convert to per-rule
    severity decisions.

    Empty map on any failure path; callers treat empty as "every rule
    keeps its configured severity" so the system never hard-fails
    because the loop can't load.
    """

    if rule_stats is None:
        if metrics_client is None:
            return RuleSeverityMap()
        try:
            from dark_factory.metrics.refinery_repository import (
                RefineryMetricsRepository,
            )
            repo = RefineryMetricsRepository(metrics_client)
            rule_stats = repo.compute_rule_override_stats(
                window_days=window_days,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_rule_severity_load_failed", error=str(exc),
            )
            return RuleSeverityMap()

    return adjust_rule_severities(rule_stats)


__all__ = [
    "RuleSeverityMap",
    "RuleSeverityOverride",
    "adjust_rule_severities",
    "apply_severity_overrides",
    "load_rule_severities",
]
