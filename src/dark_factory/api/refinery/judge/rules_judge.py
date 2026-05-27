"""RulesJudge — Phase-A deterministic gate over the rule catalog.

Runs every registered rule that isn't in the operator's disabled list.
Per-rule failures are caught so a broken rule doesn't take down the
engine — the debate continues with ``rules_engine_failed=True`` only
when zero rules succeed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from dark_factory.api.refinery.contracts import (
    CritiqueDimension,
    Draft,
    RoleContext,
    RuleViolation,
    Severity,
)
from dark_factory.api.refinery.judge.rules import RULES, RuleSpec
from dark_factory.log import trace_methods
from dark_factory.metrics.prometheus import (
    observe_refinery_rule_violation,
    observe_refinery_rules_engine_failure,
)

log = structlog.get_logger()


@dataclass
class RuleResult:
    """Phase-A output — flat list of violations, plus per-rule metadata
    the composition layer needs to decide whether to short-circuit Phase
    B and to emit counters + traces."""

    violations: list[RuleViolation] = field(default_factory=list)
    rules_run: list[str] = field(default_factory=list)
    rules_skipped: list[str] = field(default_factory=list)
    engine_failed: bool = False

    @property
    def blockers(self) -> list[RuleViolation]:
        return [v for v in self.violations if v.severity == Severity.BLOCKER]

    @property
    def warnings(self) -> list[RuleViolation]:
        return [v for v in self.violations if v.severity == Severity.WARNING]


@trace_methods
class RulesJudge:
    """Runs the rule catalog against a Draft + RoleContext."""

    def __init__(
        self,
        *,
        disabled: list[str] | None = None,
        extra_rules: list[RuleSpec] | None = None,
        dimension_overrides: dict[str, str] | None = None,
    ) -> None:
        self._disabled = set(disabled or [])
        self._extra = list(extra_rules or [])
        self._dim_overrides = dimension_overrides or {}

    @property
    def rules(self) -> list[RuleSpec]:
        """All active rules (built-ins + extras minus disabled)."""

        all_rules = [*RULES, *self._extra]
        return [r for r in all_rules if r.rule_id not in self._disabled]

    def validate(
        self,
        draft: Draft,
        context: RoleContext,
        trace: Any = None,
    ) -> RuleResult:
        result = RuleResult()
        result.rules_skipped.extend(self._disabled)
        rules_that_ran_ok = 0

        for spec in self.rules:
            try:
                violations = spec.fn(draft, context, trace)
            except Exception as exc:
                log.warning(
                    "refinery_rule_crashed",
                    rule_id=spec.rule_id, error=str(exc),
                )
                result.rules_skipped.append(f"{spec.rule_id}:error={exc}")
                continue

            rules_that_ran_ok += 1
            result.rules_run.append(spec.rule_id)

            for v in violations:
                # Apply dimension override if operator remapped this rule.
                override = self._dim_overrides.get(spec.rule_id)
                if override:
                    try:
                        v = v.model_copy(
                            update={"dimension": CritiqueDimension(override)},
                        )
                    except ValueError:
                        pass  # unknown dimension — keep the original
                result.violations.append(v)
                observe_refinery_rule_violation(
                    rule_id=v.rule_id,
                    severity=v.severity.value,
                    dimension=v.dimension.value,
                )

        if rules_that_ran_ok == 0 and self.rules:
            # Every rule crashed — mark engine failed so the composition
            # layer can choose to abstain or fall through to LLM only.
            result.engine_failed = True
            observe_refinery_rules_engine_failure()

        return result
