"""Multi-pass set-level Judge.

Where Phase 3's first-pass cross-set review is a single Judge call,
this module wraps that call in an iterated review-and-rebut loop:

1. **Round 0** — Judge.review_set produces a draft ``CrossReviewReport``.
2. **Critique pass** — each adversarial seat challenges the draft from
   its dimension at *set scope* (Engineering on feasibility of the
   cross-req patches, Security on the risk-coverage of the unresolved
   debates, etc).
3. **Ratify pass** — Judge folds the critiques into a revised
   ``CrossReviewReport`` (or accepts the draft when no critic
   surfaces a blocker).
4. **Cap** — bounded by ``refinery_set_review_max_rounds``. Default
   1 means "single shot, equivalent to the prior behaviour"; ramping
   the cap up is an opt-in operator decision.

Convergence criterion is intentionally simple in V1: when no critic
returns a blocker on the current draft, we stop. Rule-based scoring
of the report itself (a "set-level evaluation gate") is a natural
follow-up but isn't load-bearing for V1.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import structlog

from dark_factory.api.refinery.contracts import (
    CrossReviewReport,
    Critique,
    CritiqueDimension,
    Draft,
    RoleContext,
    Severity,
)
from dark_factory.api.refinery.roles.registry import RoleRegistry

log = structlog.get_logger()


_SET_LEVEL_CRITICS = ("engineering", "security", "operations", "cost")


def run_set_review(
    *,
    refined_set: list[Draft],
    run_context: dict[str, Any] | None,
    registry: RoleRegistry,
    max_rounds: int = 1,
    base_model: str | None = None,
    reasoning_effort: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> CrossReviewReport:
    """Run the multi-pass set-level Judge debate.

    Returns the converged (or final-best-effort) ``CrossReviewReport``.
    Critic exceptions are caught and logged so a single failed seat
    can't abort the whole set review.
    """

    if not registry.has("judge"):
        log.warning("set_review_no_judge")
        return CrossReviewReport()

    judge = registry.get("judge").configure(
        model=base_model, reasoning_effort=reasoning_effort,
    )
    run_context_dict = run_context or {}

    # Round 0 — Judge proposes the initial report. This single call
    # is identical to the first-pass cross-set review the orchestrator
    # used to invoke directly.
    if on_progress:
        on_progress({
            "phase": "reconciling",
            "step": "set_review_proposal",
            "round": 0,
        })
    try:
        report = judge.review_set(
            refined_set=refined_set,
            traces=[],
            run_context=run_context_dict,
        )
    except Exception as exc:
        log.warning("set_review_initial_failed", error=str(exc))
        return CrossReviewReport(
            summary=f"set-level review failed at round 0: {exc}",
        )

    # Bounded critique-and-ratify loop. ``max_rounds=1`` means the
    # initial proposal IS the final report (equivalent to prior
    # single-shot behaviour); ``max_rounds=2`` means one critique +
    # one ratify pass, etc.
    rounds_completed = 0
    while rounds_completed + 1 < max_rounds:
        critiques = _run_set_critics(
            report=report,
            refined_set=refined_set,
            run_context=run_context_dict,
            registry=registry,
            base_model=base_model,
            reasoning_effort=reasoning_effort,
            round_number=rounds_completed + 1,
            on_progress=on_progress,
        )
        blockers = [c for c in critiques if c.severity == Severity.BLOCKER]
        if not blockers:
            # No blocker — ratify the existing report.
            if on_progress:
                on_progress({
                    "phase": "reconciling",
                    "step": "set_review_no_blockers",
                    "round": rounds_completed + 1,
                })
            break

        # At least one critic raised a set-level blocker. Re-invoke
        # Judge.review_set with the critiques folded into the
        # ``traces`` channel; the prompt formatter renders them as a
        # "Critics challenged your prior draft" block so the Judge
        # revises against the panel's findings instead of re-running
        # blind on the same input.
        if on_progress:
            on_progress({
                "phase": "reconciling",
                "step": "set_review_rebut",
                "round": rounds_completed + 1,
                "blockers": len(blockers),
            })
        try:
            report = judge.review_set(
                refined_set=refined_set,
                traces=[
                    {
                        "role": c.author_role,
                        "severity": c.severity.value,
                        "dimension": c.dimension.value,
                        "finding": c.finding,
                        "proposed_fix": c.proposed_fix,
                    }
                    for c in critiques
                ],
                run_context=run_context_dict,
            )
        except Exception as exc:
            log.warning(
                "set_review_rebut_failed",
                round_number=rounds_completed + 1, error=str(exc),
            )
            break
        rounds_completed += 1

    if on_progress:
        on_progress({
            "phase": "reconciling",
            "step": "set_review_complete",
            "rounds": rounds_completed + 1,
            "summary_length": len(report.summary or ""),
        })
    return report


def _run_set_critics(
    *,
    report: CrossReviewReport,
    refined_set: list[Draft],
    run_context: dict[str, Any],
    registry: RoleRegistry,
    base_model: str | None,
    reasoning_effort: str | None,
    round_number: int,
    on_progress: Callable[[dict], None] | None = None,
) -> list[Critique]:
    """Invoke each set-level critic on the current draft report.

    Each critic gets a synthetic ``Draft`` representing the report
    itself + the refined set as evidence. Set-scope is signalled by
    ``RoleContext.evidence['set_review_report']`` so critics can
    distinguish set-level from per-requirement runs (today they
    don't branch on this; the field is the seam for future
    set-specific prompts).
    """

    out: list[Critique] = []

    # Synthetic draft: the report's summary + a digest of the patches
    # is what the critic critiques. Description packs the key counts
    # so the prompt has concrete material to push back on.
    synthetic_draft = Draft(
        requirement_id=f"set-review-r{round_number}",
        title=f"Set-level review (round {round_number})",
        description=(
            (report.summary or "(no summary)")
            + f"\n\nDuplicates: {len(report.duplicate_pairs)} | "
            + f"Coherence issues: {len(report.coherence_issues)} | "
            + f"Relationship fixes: {len(report.relationship_fixes)} | "
            + f"Priority changes: {len(report.priority_changes)} | "
            + f"Unresolved debates: {len(report.unresolved_debates)}"
        ),
        priority="high",
        tags=["set-review"],
        suggested_specs=[],
        relationships=[],
        produced_by="set-review",
        iteration=round_number,
    )

    active_roles = [r for r in _SET_LEVEL_CRITICS if registry.has(r)]
    if not active_roles:
        return out

    def _run_one(role_name: str) -> Critique:
        role = registry.get(role_name).configure(
            model=base_model, reasoning_effort=reasoning_effort,
        )
        context = RoleContext(
            role=role_name,
            requirement_id=synthetic_draft.requirement_id,
            round_number=round_number,
            evidence={
                "set_review_report": report.model_dump(),
                "refined_set": [d.model_dump() for d in refined_set],
                "run_context": run_context,
                "scope": "set",
            },
        )
        if on_progress:
            on_progress({
                "phase": "reconciling",
                "step": "set_review_critic_started",
                "round": round_number,
                "role": role_name,
            })
        try:
            critique = role.critique(synthetic_draft, context)
        except Exception as exc:
            log.warning(
                "set_review_critic_failed",
                role=role_name, error=str(exc),
            )
            critique = Critique(
                author_role=role_name,
                severity=Severity.INFO,
                dimension=CritiqueDimension.FEASIBILITY,
                finding=f"set-level critique failed: {exc}",
                proposed_fix="",
            )
        if on_progress:
            on_progress({
                "phase": "reconciling",
                "step": "set_review_critic_ready",
                "round": round_number,
                "role": role_name,
                "severity": critique.severity.value,
                "dimension": critique.dimension.value,
            })
        return critique

    with ThreadPoolExecutor(max_workers=len(active_roles)) as ex:
        # Preserve the canonical role ordering in the output even though
        # critics ran concurrently — the result list is small (≤4)
        # so the deterministic ordering is worth more than streaming.
        results = list(ex.map(_run_one, active_roles))
    out.extend(results)
    return out


__all__ = ["run_set_review"]
