"""Rule catalog for the Phase-A deterministic gate.

Every rule is a small composable function with:

- ``rule_id``       — stable identifier (used by metrics + disable list)
- ``dimension``     — which ``CritiqueDimension`` the rule informs
- ``severity``      — "blocker" | "warning"
- body signature: ``(draft, context, trace) -> list[RuleViolation]``

Rules are registered in ``RULES`` at module scope so ``RulesJudge`` can
iterate through them without reflection. Operators disable specific
rules via ``PipelineConfig.refinery_rules_disabled`` and remap
dimensions via ``refinery_rules_dimension_overrides``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    CritiqueDimension,
    Draft,
    RoleContext,
    RuleViolation,
    Severity,
)
from dark_factory.api.refinery.judge._lexicon import (
    GIVEN_WHEN_RE,
    IRREVERSIBLE_SIGNALS,
    MEASURE_RE as _MEASURE_RE,
    ROLLBACK_SIGNALS,
    VAGUE_RE as _VAGUE_RE,
    WHEN_THEN_RE,
)

RuleFn = Callable[[Draft, RoleContext, Any], list[RuleViolation]]


@dataclass(frozen=True)
class RuleSpec:
    """Metadata for one rule."""

    rule_id: str
    dimension: CritiqueDimension
    severity: Severity
    fn: RuleFn


# ── Rule bodies ───────────────────────────────────────────────────────


def _rule_ids_preserved(
    draft: Draft, context: RoleContext, trace: Any,
) -> list[RuleViolation]:
    if draft.requirement_id and draft.requirement_id != context.requirement_id:
        return [RuleViolation(
            rule_id="rule_ids_preserved",
            severity=Severity.BLOCKER,
            dimension=CritiqueDimension.COMPLETENESS,
            finding=(
                f"Draft.requirement_id='{draft.requirement_id}' does not match "
                f"RoleContext.requirement_id='{context.requirement_id}'"
            ),
            suggested_fix="Preserve the original ID so IMPLEMENTS edges stay intact",
        )]
    return []


def _rule_priority_valid(
    draft: Draft, context: RoleContext, trace: Any,
) -> list[RuleViolation]:
    if draft.priority not in ("low", "medium", "high", "critical"):
        return [RuleViolation(
            rule_id="rule_priority_valid",
            severity=Severity.BLOCKER,
            dimension=CritiqueDimension.FEASIBILITY,
            finding=f"priority='{draft.priority}' is not one of low|medium|high|critical",
            suggested_fix="Set priority to one of the allowed values",
        )]
    return []


def _rule_convergence_consistency(
    draft: Draft, context: RoleContext, trace: Any,
) -> list[RuleViolation]:
    status = draft.convergence_status
    if status is None:
        # Pre-judge drafts have no status yet — nothing to check.
        return []
    if status == ConvergenceStatus.CONVERGED:
        if draft.unresolved_points or draft.open_questions:
            return [RuleViolation(
                rule_id="rule_convergence_consistency",
                severity=Severity.BLOCKER,
                dimension=CritiqueDimension.CLARITY,
                finding=(
                    "Draft declares convergence_status=converged but still "
                    "carries unresolved_points or open_questions"
                ),
                suggested_fix="Either downgrade to short_circuited or clear the unresolved fields",
            )]
    if status == ConvergenceStatus.SHORT_CIRCUITED:
        if not (draft.unresolved_points or draft.open_questions):
            return [RuleViolation(
                rule_id="rule_convergence_consistency",
                severity=Severity.BLOCKER,
                dimension=CritiqueDimension.CLARITY,
                finding=(
                    "Draft declares short_circuited but carries no "
                    "unresolved_points nor open_questions"
                ),
                suggested_fix="Document what couldn't be resolved before short-circuiting",
            )]
    return []


def _rule_acceptance_criteria_min_count(
    draft: Draft, context: RoleContext, trace: Any,
) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    for idx, spec in enumerate(draft.suggested_specs):
        if len(spec.acceptance_criteria) < 3:
            violations.append(RuleViolation(
                rule_id="rule_acceptance_criteria_min_count",
                severity=Severity.BLOCKER,
                dimension=CritiqueDimension.TESTABILITY,
                finding=(
                    f"Spec #{idx} ({spec.title!r}) has "
                    f"{len(spec.acceptance_criteria)} acceptance criteria; "
                    "need at least 3 to demonstrate coverage"
                ),
                suggested_fix="Add at least 3 testable acceptance criteria per spec",
            ))
    return violations


def _rule_acceptance_criteria_measurable(
    draft: Draft, context: RoleContext, trace: Any,
) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    for spec in draft.suggested_specs:
        for idx, crit in enumerate(spec.acceptance_criteria):
            if _is_criterion_measurable(crit):
                continue
            violations.append(RuleViolation(
                rule_id="rule_acceptance_criteria_measurable",
                severity=Severity.BLOCKER,
                dimension=CritiqueDimension.TESTABILITY,
                finding=(
                    f"Spec {spec.title!r} criterion #{idx}: "
                    f"'{crit[:100]}' is not measurable (no WHEN/THEN shape "
                    "and no numeric+unit measure)"
                ),
                suggested_fix=(
                    "Rewrite as 'GIVEN ... WHEN ... THEN ...' with observable "
                    "conditions, or add a concrete numeric measure + unit"
                ),
            ))
    return violations


def _rule_description_specificity(
    draft: Draft, context: RoleContext, trace: Any,
) -> list[RuleViolation]:
    """Flag vague modifiers that aren't paired with a concrete measure
    within ±15 tokens. Runs on the description + title concatenated.

    Zero-length descriptions are exempt — ``rule_acceptance_criteria_*``
    will catch those from a different angle.
    """

    text = f"{draft.title}\n{draft.description}"
    if not text.strip():
        return []
    violations: list[RuleViolation] = []
    tokens = text.split()
    for match in _VAGUE_RE.finditer(text):
        word = match.group(1)
        # Find token index near the match
        token_ix = _token_index_at(tokens, text, match.start())
        # Window of 15 tokens on each side
        start = max(0, token_ix - 15)
        end = min(len(tokens), token_ix + 16)
        window = " ".join(tokens[start:end])
        if _MEASURE_RE.search(window):
            continue
        violations.append(RuleViolation(
            rule_id="rule_description_specificity",
            severity=Severity.WARNING,
            dimension=CritiqueDimension.CLARITY,
            finding=(
                f"Description uses vague modifier '{word}' without a "
                "concrete measure nearby"
            ),
            suggested_fix=(
                f"Replace '{word}' with a measurable criterion "
                "(e.g. '< 300ms', '99.9% uptime', 'handles 100 req/s')"
            ),
        ))
        # One violation per vague word per rule run — avoid spammy
        # outputs for drafts that casually use 'fast' twice.
        break
    return violations


def _rule_relationships_non_circular(
    draft: Draft, context: RoleContext, trace: Any,
) -> list[RuleViolation]:
    # Can only detect self-loops from a single draft — cross-draft cycle
    # detection happens in CrossReviewReport's relationship_fixes.
    for rel in draft.relationships:
        if rel.target_id == draft.requirement_id:
            return [RuleViolation(
                rule_id="rule_relationships_non_circular",
                severity=Severity.BLOCKER,
                dimension=CritiqueDimension.COMPLETENESS,
                finding=(
                    f"Requirement declares a self-referential '{rel.type}' "
                    f"relationship to itself"
                ),
                suggested_fix="Remove the self-referential edge",
            )]
    return []


def _rule_no_self_reference(
    draft: Draft, context: RoleContext, trace: Any,
) -> list[RuleViolation]:
    # Already covered by _rule_relationships_non_circular at the draft
    # level; kept as a separate rule_id so operators can disable one
    # without the other.
    return []


def _rule_tradeoffs_required(
    draft: Draft, context: RoleContext, trace: Any,
) -> list[RuleViolation]:
    """If the debate saw conflicting critiques that were rejected, the
    revised draft must declare explicit_tradeoffs. Without the trace
    we can't tell whether critiques were conflicting, so this rule
    only fires when the draft iteration > 0 (i.e. it's post-synthesis)
    and the Rebuttal ledger has rejected entries — trace inspection
    happens in composition.py."""

    # Phase 7 minimal: only flag when the draft is a synthesis output
    # (produced_by='judge', iteration>=1) and explicit_tradeoffs is
    # empty. The composition layer upgrades severity when it sees
    # actual conflicting Rebuttal entries in the trace.
    if draft.produced_by != "judge" or draft.iteration < 1:
        return []
    if draft.explicit_tradeoffs:
        return []
    # This is only a warning absent the trace context; composition.py
    # upgrades it to blocker when it has evidence of conflicts.
    return [RuleViolation(
        rule_id="rule_tradeoffs_required",
        severity=Severity.WARNING,
        dimension=CritiqueDimension.FEASIBILITY,
        finding=(
            "Synthesized draft has no explicit_tradeoffs declared; the "
            "panel may have glossed over a disagreement"
        ),
        suggested_fix=(
            "If the panel reconciled conflicting critiques, name the "
            "tradeoff in explicit_tradeoffs"
        ),
    )]


def _rule_reversibility_signals(
    draft: Draft, context: RoleContext, trace: Any,
) -> list[RuleViolation]:
    """Phase A signal for the REVERSIBILITY dimension. Fires WARNING
    when the draft mentions irreversible-by-default operations
    (destructive migrations, public-API contracts, regulatory
    commitments) without any rollback / feature-flag / versioning
    vocabulary. Severity escalates to BLOCKER when the draft is
    priority=critical and still names no rollback strategy — that's
    exactly the case where blast radius matters most."""

    text = f"{draft.title} {draft.description}".lower()
    irreversible_hits = [s for s in IRREVERSIBLE_SIGNALS if s in text]
    if not irreversible_hits:
        return []
    rollback_hits = [s for s in ROLLBACK_SIGNALS if s in text]
    if rollback_hits:
        return []
    severity = (
        Severity.BLOCKER if draft.priority == "critical" else Severity.WARNING
    )
    return [RuleViolation(
        rule_id="rule_reversibility_signals",
        severity=severity,
        dimension=CritiqueDimension.REVERSIBILITY,
        finding=(
            f"Draft mentions irreversible operations ({', '.join(irreversible_hits)}) "
            f"with no rollback strategy named"
        ),
        suggested_fix=(
            "Name the rollback path: feature-flag the change, version the "
            "API surface, or use additive migrations + soft-delete so the "
            "operation is reversible if it turns out wrong"
        ),
    )]


# ── Rule registry ─────────────────────────────────────────────────────


RULES: tuple[RuleSpec, ...] = (
    RuleSpec("rule_ids_preserved", CritiqueDimension.COMPLETENESS,
             Severity.BLOCKER, _rule_ids_preserved),
    RuleSpec("rule_priority_valid", CritiqueDimension.FEASIBILITY,
             Severity.BLOCKER, _rule_priority_valid),
    RuleSpec("rule_convergence_consistency", CritiqueDimension.CLARITY,
             Severity.BLOCKER, _rule_convergence_consistency),
    RuleSpec("rule_acceptance_criteria_min_count", CritiqueDimension.TESTABILITY,
             Severity.BLOCKER, _rule_acceptance_criteria_min_count),
    RuleSpec("rule_acceptance_criteria_measurable", CritiqueDimension.TESTABILITY,
             Severity.BLOCKER, _rule_acceptance_criteria_measurable),
    RuleSpec("rule_description_specificity", CritiqueDimension.CLARITY,
             Severity.WARNING, _rule_description_specificity),
    RuleSpec("rule_relationships_non_circular", CritiqueDimension.COMPLETENESS,
             Severity.BLOCKER, _rule_relationships_non_circular),
    RuleSpec("rule_no_self_reference", CritiqueDimension.COMPLETENESS,
             Severity.BLOCKER, _rule_no_self_reference),
    RuleSpec("rule_tradeoffs_required", CritiqueDimension.FEASIBILITY,
             Severity.WARNING, _rule_tradeoffs_required),
    RuleSpec("rule_reversibility_signals", CritiqueDimension.REVERSIBILITY,
             Severity.WARNING, _rule_reversibility_signals),
)


# ── Helpers ───────────────────────────────────────────────────────────


def _is_criterion_measurable(text: str) -> bool:
    """Return True if *text* matches a WHEN/THEN or GIVEN/WHEN/THEN
    shape, or contains a numeric measure with unit."""

    if not text or not text.strip():
        return False
    if WHEN_THEN_RE.search(text):
        return True
    if GIVEN_WHEN_RE.search(text):
        return True
    if _MEASURE_RE.search(text):
        return True
    # A criterion that declares a boolean outcome via an observable
    # verb (returns 4xx, rejects empty input, logs an error) is also
    # measurable — we accept either modal + verb OR bare third-person
    # present-tense verb. This covers both "must reject empty input"
    # and "rejects empty input".
    observable_verbs = (
        r"(?:return|reject|emit|produce|log|block|deny|allow|fail|"
        r"succeed|raise|respond|redirect|expire|persist|publish|"
        r"enforce|validate|match)"
    )
    if re.search(
        rf"(?:must|should|will)\s+{observable_verbs}\b",
        text,
        re.IGNORECASE,
    ):
        return True
    # Bare third-person verb form (common in spec prose).
    if re.search(rf"\b{observable_verbs}s?\b", text, re.IGNORECASE):
        return True
    return False


def _token_index_at(tokens: list[str], text: str, char_index: int) -> int:
    """Given a char_index into *text*, return the token index in the
    naively-split token list. Used by rule_description_specificity to
    find a token window around a vague modifier."""

    cursor = 0
    for i, tok in enumerate(tokens):
        idx = text.find(tok, cursor)
        if idx < 0:
            continue
        cursor = idx + len(tok)
        if idx >= char_index:
            return i
    return len(tokens) - 1
