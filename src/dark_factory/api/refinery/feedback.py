"""Active-learning feedback loop.

When the operator dismisses or accepts a suggested memory in the UI,
``record_memory_feedback`` does three things in best-effort order:

1. Persist the decision in ``refinery_memory_feedback`` (Postgres) as
   a forensic record of operator follow-through on suggestions. This
   table is read by ad-hoc analytics queries, not by any in-process
   aggregator today; no learning loop currently consumes it. (Future
   work could join it against the existing role-weighting signal so
   memory dismissals attenuate the suggesting role; that link does
   not exist yet.)
2. Adjust the memory's ``relevance_score`` in Neo4j (boost on accept,
   demote on dismiss) — this is the load-bearing learning signal:
   future role retrieval re-ranks future debates accordingly.
3. On a dismissal with a ``reason``, emit a ``Conflict`` memory tagged
   ``cause="user_override"`` so the next debate sees the operator's
   prior judgement as institutional context.

Every step degrades gracefully — missing repo, missing memory id,
missing label all become no-ops with structured warnings. The UI
action never fails because of a feedback-loop hiccup.
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()


from dark_factory.memory.repository import MEMORY_KIND_TO_LABEL

_DECISION_VALUES = frozenset({"accepted", "dismissed", "edited"})

_BOOST_DELTA = 0.10
_DEMOTE_DELTA = 0.05


def record_memory_feedback(
    *,
    memory_id: str,
    memory_kind: str,
    decision: str,
    metrics_repo: Any | None = None,
    memory_repo: Any | None = None,
    refinery_run_id: str | None = None,
    source_role: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Record one operator feedback decision and apply its consequences.

    Returns a dict describing what happened so the caller (the route
    handler) can echo it back to the UI for debugging. Never raises.
    """

    if decision not in _DECISION_VALUES:
        return {
            "ok": False,
            "reason": f"invalid decision '{decision}'; must be one of "
                       f"{sorted(_DECISION_VALUES)}",
        }

    label = MEMORY_KIND_TO_LABEL.get(memory_kind)
    if label is None:
        return {
            "ok": False,
            "reason": f"unknown memory_kind '{memory_kind}'",
        }

    result: dict[str, Any] = {
        "ok": True,
        "memory_id": memory_id,
        "decision": decision,
        "telemetry_recorded": False,
        "relevance_adjusted": False,
        "conflict_emitted": False,
    }

    # 1. Postgres telemetry — feeds role weighting and historical analysis.
    if metrics_repo is not None:
        try:
            metrics_repo.record_memory_feedback(
                memory_id=memory_id,
                memory_kind=memory_kind,
                decision=decision,
                refinery_run_id=refinery_run_id,
                source_role=source_role,
                reason=reason,
            )
            result["telemetry_recorded"] = True
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_feedback_telemetry_failed",
                memory_id=memory_id, error=str(exc),
            )

    # 2. Neo4j relevance score — re-ranks future retrieval.
    if memory_repo is not None:
        try:
            if decision == "accepted":
                memory_repo.boost_relevance(
                    memory_id, label, delta=_BOOST_DELTA,
                )
                result["relevance_adjusted"] = True
            elif decision == "dismissed":
                memory_repo.demote_relevance(
                    memory_id, label, delta=_DEMOTE_DELTA,
                )
                result["relevance_adjusted"] = True
            # "edited" — no relevance adjustment; the edit itself is
            # the signal and the repo's edit path handles persistence.
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_feedback_relevance_failed",
                memory_id=memory_id, label=label, error=str(exc),
            )

    # 3. Conflict memory — only on a *reasoned* dismissal so we don't
    #    flood storage with empty overrides.
    if (
        memory_repo is not None
        and decision == "dismissed"
        and reason
        and reason.strip()
    ):
        try:
            memory_repo.record_conflict(
                summary=f"Operator dismissed {memory_kind} memory",
                body=reason.strip(),
                conflict_parties=["user", source_role or "system"],
                conflict_resolution=f"Memory {memory_id} demoted",
                cause="user_override",
                provenance_refinery_run_id=refinery_run_id or "",
            )
            result["conflict_emitted"] = True
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "refinery_feedback_conflict_failed",
                memory_id=memory_id, error=str(exc),
            )

    return result


__all__ = ["record_memory_feedback"]
