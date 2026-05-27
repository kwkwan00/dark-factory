"""apply_cross_review_report — pure structural patcher.

No LLM dependency. Takes a ``CrossReviewReport`` (``ReconciliationReport``
is a strict subset with the same field names and is also accepted) and
a list of ``RefinedRequirement``; returns the patched list with
duplicates marked, relationship fixes applied, and priority changes
recorded. The PATCH endpoint commits whatever this returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from dark_factory.api.refinery.models import (
    RefinedRequirement,
    RequirementRelationship,
)

if TYPE_CHECKING:
    from dark_factory.api.refinery.contracts import CrossReviewReport
    from dark_factory.api.refinery.models import ReconciliationReport


class _ReportLike(Protocol):
    """Minimal protocol CrossReviewReport / ReconciliationReport both satisfy."""

    duplicate_pairs: list[dict]
    relationship_fixes: list[dict]
    priority_changes: list[dict]


def apply_cross_review_report(
    refined: list[RefinedRequirement],
    report: "CrossReviewReport | ReconciliationReport",
) -> list[RefinedRequirement]:
    """Apply the patcher logic and return the updated requirement list.

    Idempotent — applying the same report twice produces the same result
    (de-duplicated relationship edges, identical change notes).

    Field-name parity with ``ReconciliationReport`` is intentional so
    callers can construct either the simpler shape or the
    debate-scoped ``CrossReviewReport`` and route through the same
    patcher.
    """

    req_map: dict[str, RefinedRequirement] = {r.id: r for r in refined}

    # Relationship fixes (add / remove edges).
    for fix in getattr(report, "relationship_fixes", []):
        src_id = fix.get("source_id", "")
        tgt_id = fix.get("target_id", "")
        action = fix.get("action", "")
        rel_type = fix.get("type", "related_to")
        rationale = fix.get("rationale", "")

        if action == "add" and src_id in req_map:
            src = req_map[src_id]
            if not any(
                r.target_id == tgt_id and r.type == rel_type
                for r in src.relationships
            ):
                src.relationships.append(RequirementRelationship(
                    target_id=tgt_id, type=rel_type, rationale=rationale,
                ))
        elif action == "remove" and src_id in req_map:
            src = req_map[src_id]
            src.relationships = [
                r for r in src.relationships
                if not (r.target_id == tgt_id and r.type == rel_type)
            ]

    # Priority changes — idempotent: no-op when the current priority
    # already matches the target, so replaying the report doesn't add
    # a new "high → high" note.
    for change in getattr(report, "priority_changes", []):
        rid = change.get("requirement_id", "")
        new_pri = change.get("new_priority", "")
        if rid not in req_map or not new_pri:
            continue
        req = req_map[rid]
        old_pri = req.priority
        if old_pri == new_pri:
            continue
        req.priority = new_pri
        note = (
            f"Priority changed from {old_pri} to {new_pri}: "
            f"{change.get('rationale', '')}"
        )
        if note not in req.changes:
            req.changes.append(note)

    # Duplicate markers.
    for dup in getattr(report, "duplicate_pairs", []):
        remove_id = dup.get("remove_id", "")
        if remove_id in req_map:
            req = req_map[remove_id]
            keep_id = dup.get("keep_id", "?")
            rationale = dup.get("rationale", "")
            dup_note = f"Potential duplicate of {keep_id}: {rationale}"
            if dup_note not in req.changes:
                req.changes.append(dup_note)
            if not any(
                r.target_id == keep_id and r.type == "replaces"
                for r in req.relationships
            ):
                req.relationships.append(RequirementRelationship(
                    target_id=keep_id,
                    type="replaces",
                    rationale=f"Duplicate: {rationale}",
                ))

    return list(req_map.values())
