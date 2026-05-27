"""Memory producers — fire at specific debate events.

Each function is a pure inspector: it takes debate artefacts (a
rebuttal, a critique, a finalized draft, etc.) and returns zero or more
``SuggestedMemoryV2`` records. Nothing is written to stores here —
producers only build the *suggested* memory payload. Write-back happens
atomically when the user applies the converged requirement to the graph
(see ``/api/graph/requirements/{req_id}`` PATCH handler in Phase 10).

Lifecycle table (from the plan):

- ``synthesize_node`` rebuttal with a rationale that generalises
  → one DECISION memory per accepted critique pair (heuristic).
- ``critic_node`` blocker citing a system / business limit
  → one CONSTRAINT memory (``constraint_domain=system|business``).
- ``critic_node`` blocker citing a runtime / incident pattern
  → one INCIDENT memory (``incident_severity`` inferred).
- ``finalize_node`` with ``disagreement_score ≥ threshold``
  → one CONFLICT memory with ``conflict_parties``.
- ``reconcile_node`` (short-circuit) — **unconditional**
  → one CONFLICT memory tagged ``cause="non_convergence"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    Critique,
    Draft,
    MemoryKind,
    Rebuttal,
    RebuttalEntry,
    Severity,
    SuggestedMemoryV2,
    ValidationStatus,
)


@dataclass
class MemoryProducerResult:
    """What one producer emits. Orchestrator aggregates these across
    rounds into ``DebateTrace.memory_audits``."""

    memories: list[SuggestedMemoryV2] = field(default_factory=list)
    skipped_reason: str = ""


# ─────────────────────────────────────────────────────────────────────
# DECISION — from synthesized rebuttals
# ─────────────────────────────────────────────────────────────────────


def produce_decision_from_rebuttal(
    rebuttal: Rebuttal,
    *,
    critiques: list[Critique],
    requirement_id: str,
    round_number: int,
    source_role: str = "judge",
    converged: bool,
) -> MemoryProducerResult:
    """Emit DECISION memories for rebuttal entries that generalise.

    Heuristic: a rebuttal entry generalises when (a) the action is
    accepted OR rejected (deferred entries are just work-in-progress),
    (b) the rationale is ≥ 25 chars (matches the no-rubber-stamp
    validator), and (c) the matching critique has conflict-indicating
    severity (warning / blocker). Trivial INFO critique rejections
    don't merit a memory.
    """

    result = MemoryProducerResult()

    crit_by_ref: dict[str, Critique] = {
        f"c-{i}": c for i, c in enumerate(critiques)
    }
    for entry in rebuttal.entries:
        if entry.action not in ("accepted", "rejected"):
            continue
        if len(entry.rationale or "") < 25:
            continue
        ref_crit = crit_by_ref.get(entry.critique_ref)
        if ref_crit is None:
            continue
        if ref_crit.severity == Severity.INFO:
            continue

        summary = (
            f"{'accepted' if entry.action == 'accepted' else 'rejected'} "
            f"{ref_crit.author_role}'s {ref_crit.dimension.value} critique"
        )[:120]
        result.memories.append(SuggestedMemoryV2(
            kind=MemoryKind.DECISION,
            summary=summary,
            body=f"{ref_crit.finding}\n\nJudge reasoning:\n{entry.rationale}",
            context=f"Round {round_number}, requirement {requirement_id}",
            source_role=source_role,
            source_requirement_id=requirement_id,
            source_debate_round=round_number,
            rationale=entry.rationale,
            decision_alternatives=[ref_crit.proposed_fix] if ref_crit.proposed_fix else None,
            validation_status=(
                ValidationStatus.PRODUCED if converged
                else ValidationStatus.UNVALIDATED
            ),
        ))
    return result


# ─────────────────────────────────────────────────────────────────────
# CONSTRAINT — from blocker critiques citing a limit
# ─────────────────────────────────────────────────────────────────────


# Whole-word (re.search with \b) keyword sets so "api" doesn't match
# "API shape". Ordering matters: business keywords are checked first
# because GDPR / SLA / compliance are more specific signals than
# generic "budget" (which appears in both lists).
_CONSTRAINT_KEYWORDS_BUSINESS = (
    r"compliance", r"gdpr", r"hipaa", r"sla", r"slo",
    r"contract", r"policy", r"licen[cs]e", r"pricing",
    r"retention",
)
_CONSTRAINT_KEYWORDS_SYSTEM = (
    r"rate[- ]?limit", r"timeout", r"quota", r"latency",
    r"throughput", r"budget", r"stack[- ]?pin", r"sdk[- ]?version",
    r"memory\s+cap", r"cpu\s+cap", r"capacity",
)

_BUSINESS_RE = re.compile(
    r"\b(?:" + "|".join(_CONSTRAINT_KEYWORDS_BUSINESS) + r")\b",
    re.IGNORECASE,
)
_SYSTEM_RE = re.compile(
    r"\b(?:" + "|".join(_CONSTRAINT_KEYWORDS_SYSTEM) + r")\b",
    re.IGNORECASE,
)


def produce_constraint_from_blocker(
    critique: Critique,
    *,
    requirement_id: str,
    round_number: int,
    converged: bool,
) -> MemoryProducerResult:
    """Emit a CONSTRAINT memory when a blocker critique cites a system
    or business limit. Non-blocker critiques and blockers without a
    limit-keyword don't produce a constraint."""

    result = MemoryProducerResult()
    if critique.severity != Severity.BLOCKER:
        result.skipped_reason = "severity != blocker"
        return result

    text = f"{critique.finding} {critique.proposed_fix}"
    business_hit = bool(_BUSINESS_RE.search(text))
    system_hit = bool(_SYSTEM_RE.search(text))
    if not (system_hit or business_hit):
        result.skipped_reason = "no limit-keyword"
        return result

    domain = "business" if business_hit else "system"
    result.memories.append(SuggestedMemoryV2(
        kind=MemoryKind.CONSTRAINT,
        summary=critique.finding[:120],
        body=f"{critique.finding}\n\nFix: {critique.proposed_fix}",
        source_role=critique.author_role,
        source_requirement_id=requirement_id,
        source_debate_round=round_number,
        rationale=f"BLOCKER critique from {critique.author_role}",
        constraint_domain=domain,  # type: ignore[arg-type]
        validation_status=(
            ValidationStatus.PRODUCED if converged
            else ValidationStatus.UNVALIDATED
        ),
    ))
    return result


# ─────────────────────────────────────────────────────────────────────
# ANTI_PATTERN — from blocker critiques citing recurring negative guidance
# ─────────────────────────────────────────────────────────────────────


_ANTI_PATTERN_KEYWORDS = (
    "anti-pattern", "antipattern", "anti pattern",
    "don't ", "do not ", "avoid ", "stop using ",
    "harmful pattern", "footgun", "smell",
)


def produce_anti_pattern_from_blocker(
    critique: Critique,
    *,
    requirement_id: str,
    round_number: int,
    converged: bool,
) -> MemoryProducerResult:
    """Emit an ANTI_PATTERN memory when a blocker critique flags a draft
    as repeating a recurring negative pattern AND names the alternative
    via ``proposed_fix``. The proposed_fix is required — an anti-pattern
    without a recommended replacement is just a complaint, not actionable
    guidance for future debates.

    Distinct from CONSTRAINT (a fixed limit) and INCIDENT (a past
    runtime failure). Anti-patterns capture *what to avoid*; future
    Product drafts read them at propose-time so they don't re-litigate
    the same rejected approach.
    """

    result = MemoryProducerResult()
    if critique.severity != Severity.BLOCKER:
        result.skipped_reason = "severity != blocker"
        return result

    text = f"{critique.finding} {critique.proposed_fix}".lower()
    if not any(kw in text for kw in _ANTI_PATTERN_KEYWORDS):
        result.skipped_reason = "no anti-pattern keyword"
        return result

    if not critique.proposed_fix.strip():
        result.skipped_reason = "no alternative named in proposed_fix"
        return result

    result.memories.append(SuggestedMemoryV2(
        kind=MemoryKind.ANTI_PATTERN,
        summary=critique.finding[:120],
        body=f"Avoid: {critique.finding}\n\nInstead: {critique.proposed_fix}",
        source_role=critique.author_role,
        source_requirement_id=requirement_id,
        source_debate_round=round_number,
        rationale=(
            f"BLOCKER critique from {critique.author_role} flagged a "
            f"recurring negative pattern with a named alternative"
        ),
        anti_pattern_alternative=critique.proposed_fix,
        anti_pattern_harm=critique.finding,
        validation_status=(
            ValidationStatus.PRODUCED if converged
            else ValidationStatus.UNVALIDATED
        ),
    ))
    return result


# ─────────────────────────────────────────────────────────────────────
# INCIDENT — from blocker critiques citing a runtime failure
# ─────────────────────────────────────────────────────────────────────


_INCIDENT_KEYWORDS = (
    "incident", "outage", "crash", "corrupt", "leak",
    "postmortem", "sev1", "sev2", "sev3", "regression",
)


def produce_incident_from_blocker(
    critique: Critique,
    *,
    requirement_id: str,
    round_number: int,
    converged: bool,
) -> MemoryProducerResult:
    """Emit an INCIDENT memory when a blocker critique cites a runtime
    failure or incident pattern. Distinct from CONSTRAINT: incidents
    are *observed past failures*, constraints are *known limits*."""

    result = MemoryProducerResult()
    if critique.severity != Severity.BLOCKER:
        result.skipped_reason = "severity != blocker"
        return result

    text = f"{critique.finding} {critique.proposed_fix}".lower()
    if not any(kw in text for kw in _INCIDENT_KEYWORDS):
        result.skipped_reason = "no incident-keyword"
        return result

    # Infer severity
    if "sev1" in text:
        incident_sev = "sev1"
    elif "sev2" in text:
        incident_sev = "sev2"
    else:
        incident_sev = "sev3"

    result.memories.append(SuggestedMemoryV2(
        kind=MemoryKind.INCIDENT,
        summary=critique.finding[:120],
        body=f"{critique.finding}\n\nFix: {critique.proposed_fix}",
        source_role=critique.author_role,
        source_requirement_id=requirement_id,
        source_debate_round=round_number,
        rationale=f"{incident_sev} incident pattern cited by {critique.author_role}",
        incident_severity=incident_sev,  # type: ignore[arg-type]
        validation_status=(
            ValidationStatus.PRODUCED if converged
            else ValidationStatus.UNVALIDATED
        ),
    ))
    return result


# ─────────────────────────────────────────────────────────────────────
# CONFLICT — from high-disagreement finalize OR short-circuit reconcile
# ─────────────────────────────────────────────────────────────────────


def produce_conflict_on_disagreement(
    *,
    disagreement_score: float,
    threshold: float,
    critiques_by_round: dict[int, list[Critique]],
    rebuttals_by_round: dict[int, Rebuttal | None],
    requirement_id: str,
    rounds_executed: int,
    converged: bool,
) -> MemoryProducerResult:
    """Emit a CONFLICT memory when the finalize round's disagreement
    score crosses the threshold. Not unconditional — that's
    ``produce_conflict_on_short_circuit``'s job."""

    result = MemoryProducerResult()
    if disagreement_score < threshold:
        result.skipped_reason = (
            f"disagreement_score={disagreement_score:.2f} < threshold={threshold:.2f}"
        )
        return result

    # Identify parties — the roles whose critiques raised WARNING or
    # BLOCKER findings that the final rebuttal didn't accept.
    parties: set[str] = set()
    for rn, crits in critiques_by_round.items():
        for c in crits:
            if c.severity in (Severity.BLOCKER, Severity.WARNING):
                parties.add(c.author_role)

    # Resolution — the final rebuttal's notes (accepted entries).
    final_reb = rebuttals_by_round.get(rounds_executed)
    resolution = ""
    if final_reb:
        accepted = [e.rationale for e in final_reb.entries if e.action == "accepted"]
        resolution = " | ".join(accepted)[:500] if accepted else ""

    result.memories.append(SuggestedMemoryV2(
        kind=MemoryKind.CONFLICT,
        summary=(
            f"Panel disagreed sharply on requirement {requirement_id} "
            f"({disagreement_score:.2f} disagreement)"
        )[:120],
        body=(
            "Parties: " + ", ".join(sorted(parties)) + "\n"
            f"Rounds executed: {rounds_executed}\n"
            f"Disagreement score: {disagreement_score:.2f}\n"
            f"Resolution: {resolution or '(none recorded)'}"
        ),
        source_role="judge",
        source_requirement_id=requirement_id,
        source_debate_round=rounds_executed,
        rationale=f"disagreement_score={disagreement_score:.2f}",
        conflict_parties=sorted(parties),
        conflict_resolution=resolution or None,
        validation_status=(
            ValidationStatus.PRODUCED if converged
            else ValidationStatus.UNVALIDATED
        ),
    ))
    return result


def produce_hypotheses_from_open_questions(
    *,
    final_draft: Draft,
    requirement_id: str,
    rounds_executed: int,
) -> MemoryProducerResult:
    """Emit one HYPOTHESIS memory per concrete open question on the
    finalized draft. Each hypothesis carries the question as its
    verification query — the next debate's Research agent treats it
    as a Librarian prompt and updates the hypothesis status to
    ``verified`` or ``refuted`` once corroborating sources exist.

    Hypotheses are produced regardless of convergence: a converged
    debate may still surface a question worth verifying, and a short-
    circuited debate's open questions are exactly the ones future
    debates need to chase. Status is always ``open`` at production
    time; the Editor / Librarian moves it forward later.
    """

    result = MemoryProducerResult()
    questions = [q.strip() for q in (final_draft.open_questions or []) if q.strip()]
    if not questions:
        result.skipped_reason = "no open questions on final draft"
        return result

    converged = final_draft.convergence_status == ConvergenceStatus.CONVERGED
    for question in questions:
        result.memories.append(SuggestedMemoryV2(
            kind=MemoryKind.HYPOTHESIS,
            summary=question[:120],
            body=(
                f"Open question surfaced during debate of "
                f"requirement {requirement_id}.\n\n"
                f"Verification target: {question}"
            ),
            source_role="judge",
            source_requirement_id=requirement_id,
            source_debate_round=rounds_executed,
            rationale=(
                "panel surfaced this as unresolved at finalize; "
                "next debate's Research should target it"
            ),
            hypothesis_verification_query=question,
            hypothesis_status="open",
            validation_status=(
                ValidationStatus.PRODUCED if converged
                else ValidationStatus.UNVALIDATED
            ),
        ))
    return result


def produce_conflict_on_short_circuit(
    *,
    final_draft: Draft,
    critiques_by_round: dict[int, list[Critique]],
    requirement_id: str,
    rounds_executed: int,
) -> MemoryProducerResult:
    """Unconditional CONFLICT memory emitted when the reconcile node
    fires. Tagged ``cause="non_convergence"`` so future debates can
    distinguish "the panel couldn't agree" from "the panel agreed after
    a hard-fought debate".

    Critical: this memory is ALWAYS validation_status=VALIDATED because
    non-convergence itself is a factual output — not subject to the
    "only save memories from converged debates" rule for other kinds.
    """

    result = MemoryProducerResult()
    if final_draft.convergence_status != ConvergenceStatus.SHORT_CIRCUITED:
        result.skipped_reason = "draft is not short_circuited"
        return result

    # Parties = roles that raised unaddressed blockers across all rounds.
    parties: set[str] = set()
    for rn, crits in critiques_by_round.items():
        for c in crits:
            if c.severity == Severity.BLOCKER:
                parties.add(c.author_role)

    result.memories.append(SuggestedMemoryV2(
        kind=MemoryKind.CONFLICT,
        summary=(
            f"Non-convergence on requirement {requirement_id} after "
            f"{rounds_executed} rounds"
        )[:120],
        body=(
            "Parties with unaddressed blockers: "
            + ", ".join(sorted(parties)) + "\n"
            f"Unresolved points:\n"
            + "\n".join(f"- {p}" for p in final_draft.unresolved_points)
            + "\nOpen questions:\n"
            + "\n".join(f"- {q}" for q in final_draft.open_questions)
        ),
        source_role="judge",
        source_requirement_id=requirement_id,
        source_debate_round=rounds_executed,
        rationale="non-convergence is itself load-bearing institutional knowledge",
        conflict_parties=sorted(parties),
        conflict_resolution=None,  # explicitly unresolved
        # Non-convergence is always validated — the reconcile step
        # itself is the panel's consolidated verdict.
        validation_status=ValidationStatus.VALIDATED,
    ))
    return result
