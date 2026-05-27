"""LangGraph StateGraph for the per-requirement debate.

Topology (see ``build_debate_graph`` below):

    START → generator → (critic fan-out) → synthesize → score
                                               ↓
    router: finalize | research | escalate | reconcile | next_round
        research → synthesize
        escalate → critic fan-out (strong model)
        reconcile → finalize (short-circuit)
        finalize → END

The graph returns a compiled ``Runnable`` whose ``invoke`` takes a
``DebateState`` and yields the terminal state. ``stream.py`` builds +
invokes the graph per requirement and forwards node-level progress to
the frontend via the ``emit`` callback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    Critique,
    CritiqueDimension,
    Draft,
    RawRequirement,
    Rebuttal,
    RoleContext,
    Severity,
)
from dark_factory.api.refinery.debate.state import DebateState, DraftRecord
from dark_factory.api.refinery.memory import (
    MemoryProducerResult,
    produce_anti_pattern_from_blocker,
    produce_conflict_on_disagreement,
    produce_conflict_on_short_circuit,
    produce_constraint_from_blocker,
    produce_decision_from_rebuttal,
    produce_hypotheses_from_open_questions,
    produce_incident_from_blocker,
)
from dark_factory.api.refinery.roles._shared.critic_base import (
    make_placeholder_critique,
)
from dark_factory.api.refinery.roles.registry import RoleRegistry
from dark_factory.config import PipelineConfig

# The four adversarial seats, in deterministic order for fan-out.
CRITIC_ROLES: tuple[str, ...] = ("engineering", "security", "operations", "cost")

# Default dimension per critic role — used to build placeholder critiques
# when the role isn't registered (dev-mode without LLM).
_CRITIC_DEFAULT_DIMENSION: dict[str, CritiqueDimension] = {
    "engineering": CritiqueDimension.FEASIBILITY,
    "security": CritiqueDimension.RISK_COVERAGE,
    "operations": CritiqueDimension.COMPLETENESS,
    "cost": CritiqueDimension.FEASIBILITY,
}

log = structlog.get_logger()


def _count_severities(critiques: list) -> tuple[int, int]:
    """Return ``(blocker_count, warning_count)`` from a list of Critique
    models OR raw dicts. Used by the synthesize / reconcile event payloads
    so both shapes resolve to the same enum-backed comparison."""

    blockers = warnings = 0
    for c in critiques:
        sev = c.severity if hasattr(c, "severity") else (c or {}).get("severity")
        sev_val = sev.value if hasattr(sev, "value") else sev
        if sev_val == Severity.BLOCKER.value:
            blockers += 1
        elif sev_val == Severity.WARNING.value:
            warnings += 1
    return blockers, warnings


# ─────────────────────────────────────────────────────────────────────
# Memory audit accumulation helper
# ─────────────────────────────────────────────────────────────────────


def _audits_from(*results: MemoryProducerResult) -> list[dict]:
    """Flatten one or more producer results into a list of
    ``MemoryAuditEntry.model_dump()`` dicts ready for the
    ``memory_audits`` reducer. Each suggested memory becomes one
    audit entry stamped ``outcome="suggested"`` — the user's
    Apply-to-Graph action later promotes selected entries to
    ``outcome="saved"`` via the PATCH handler."""

    from uuid import uuid4

    from dark_factory.api.refinery.observability.trace import MemoryAuditEntry

    audits: list[dict] = []
    for result in results:
        for mem in result.memories:
            audit = MemoryAuditEntry(
                suggested_memory_id=f"mem-{uuid4().hex[:8]}",
                kind=mem.kind,
                source_role=mem.source_role,
                summary=mem.summary,
                validation_status=mem.validation_status,
                outcome="suggested",
                provenance_source_tier_mix=mem.provenance_source_tier_mix,
                provenance_confidence=mem.provenance_confidence,
            )
            audits.append(audit.model_dump(mode="json"))
    return audits


# ─────────────────────────────────────────────────────────────────────
# Node factories — closures over (registry, emit) produce plain state→state functions
# ─────────────────────────────────────────────────────────────────────


def _make_generator_node(
    registry: RoleRegistry,
    emit: Callable[[dict], None] | None = None,
) -> Callable[[DebateState], dict]:
    """Generator node — runs ``ProductRole.propose``."""

    def generator_node(state: DebateState) -> dict:
        requirement = RawRequirement.model_validate(state["requirement"])
        if emit:
            emit({"phase": "refining", "debate_event": "generator_started",
                  "requirement_id": requirement.id,
                  "title": (requirement.title or "")[:80],
                  "description_preview": (requirement.description or "")[:240],
                  "priority": requirement.priority,
                  "tags": list(requirement.tags or [])[:8]})

        # Override instance model/effort to whatever the outer orchestrator
        # resolved — the caller's base_model, or strong_model after escalation.
        product = registry.get("product").configure(
            model=state.get("base_model"),
            reasoning_effort=state.get("reasoning_effort"),
        )

        context = RoleContext(
            role="product",
            requirement_id=requirement.id,
            round_number=0,
            evidence=state.get("evidence_bag", {}),
        )
        try:
            draft = product.propose(requirement, context)
        except Exception as exc:
            log.warning("refinery_generator_failed", error=str(exc))
            return {
                "errors": [{"node": "generator", "round": 0, "message": str(exc)}],
                "aborted_reason": f"generator failed: {exc}",
            }

        record: DraftRecord = {
            "round": 0,
            "refined": draft.model_dump(),
            "author_role": "product",
            "model_tier": state.get("base_model", ""),
        }
        if emit:
            emit({"phase": "refining", "debate_event": "draft_ready", "round": 0,
                  "requirement_id": requirement.id,
                  "title": (draft.title or "")[:80],
                  "description_preview": (draft.description or "")[:240],
                  "priority": draft.priority,
                  "specs_count": len(draft.suggested_specs),
                  "relationships_count": len(draft.relationships)})
        return {"draft_history": [record], "round_number": 0}

    return generator_node


def _make_critic_node(
    registry: RoleRegistry,
    emit: Callable[[dict], None] | None = None,
) -> Callable[[DebateState], dict]:
    """Critic node — runs once per role per round via LangGraph's
    ``Send`` fan-out. The role being invoked is carried on the
    ``_critic_role`` key that ``dispatch_critics`` stamps on each
    Send branch's state overlay."""

    def critic_node(state: DebateState) -> dict:
        role_name = state.get("_critic_role") or ""  # type: ignore[typeddict-item]
        history = state.get("draft_history", [])
        if not history:
            return {
                "errors": [{"node": f"critic.{role_name}", "round": 0,
                            "message": "critic invoked without a draft"}],
            }

        draft = Draft.model_validate(history[-1]["refined"])
        round_num = state.get("round_number", 0) + 1
        requirement_id = state["requirement"].get("id", "?")
        if emit:
            emit({"phase": "refining", "debate_event": "critic_started",
                  "role": role_name, "round": round_num,
                  "requirement_id": requirement_id,
                  "target_title": (draft.title or "")[:80],
                  "target_priority": draft.priority,
                  "target_iteration": draft.iteration})

        default_dim = _CRITIC_DEFAULT_DIMENSION.get(
            role_name, CritiqueDimension.FEASIBILITY,
        )

        # If the critic isn't registered (e.g. tests that use stub
        # registries) or its LLM binding fails, fall back to a
        # placeholder so the round continues.
        if not registry.has(role_name):
            placeholder = make_placeholder_critique(
                role_name=role_name,
                dimension=default_dim,
                reason="role not registered",
                retry_count=0,
                error="role-not-registered",
            )
            if emit:
                emit({"phase": "refining", "debate_event": "critic_placeholder",
                      "role": role_name, "round": round_num,
                      "requirement_id": requirement_id,
                      "reason": "role not registered",
                      "severity": placeholder.severity.value,
                      "dimension": placeholder.dimension.value})
            return {"critiques_by_round": {round_num: [placeholder.model_dump()]}}

        role = registry.get(role_name).configure(
            model=state.get("base_model"),
            reasoning_effort=state.get("reasoning_effort"),
        )

        # Cross-debate coordination — fetch findings from concurrent
        # sibling debates (different requirement_id, same
        # refinery_run_id) and inject into the role's evidence bag.
        # The shared critic base renders these as a prompt prefix.
        evidence = dict(state.get("evidence_bag", {}) or {})
        from dark_factory.api.refinery.debate.cross_debate_bus import (
            get_global_bus,
        )
        bus = evidence.get("cross_debate_bus") or get_global_bus()
        run_id = state.get("refinery_run_id") or ""
        if run_id:
            findings = bus.read_for(
                refinery_run_id=run_id,
                requirement_id=requirement_id,
                kinds=["critic_blocker", "judge_tradeoff", "constraint"],
                limit=8,
            )
            if findings:
                evidence["cross_debate_findings"] = [
                    {
                        "requirement_id": f.requirement_id,
                        "kind": f.kind,
                        "body": f.body,
                        "round_number": f.round_number,
                    }
                    for f in findings
                ]

        context = RoleContext(
            role=role_name,
            requirement_id=requirement_id,
            round_number=round_num,
            evidence=evidence,
        )

        try:
            critique = role.critique(draft, context)
        except Exception as exc:
            log.warning(
                "refinery_critic_node_failed",
                role=role_name, error=str(exc),
            )
            critique = make_placeholder_critique(
                role_name=role_name,
                dimension=default_dim,
                reason=str(exc),
                retry_count=2,
                error=str(exc),
            )

        # Post BLOCKER findings back to the bus so concurrent siblings
        # see this requirement's hard constraints in their next round.
        # Warnings + INFO stay local — they're noisier and less
        # actionable for sibling debates.
        if run_id and critique.severity == Severity.BLOCKER and not critique.error:
            try:
                bus.post(
                    refinery_run_id=run_id,
                    requirement_id=requirement_id,
                    kind="critic_blocker",
                    body={
                        "role": role_name,
                        "severity": critique.severity.value,
                        "dimension": critique.dimension.value,
                        "finding": critique.finding,
                        "proposed_fix": critique.proposed_fix,
                    },
                    round_number=round_num,
                )
            except Exception:  # pragma: no cover — defensive
                log.exception("cross_debate_bus_post_failed")

        if emit:
            emit({"phase": "refining", "debate_event": "critic_ready",
                  "role": role_name, "round": round_num,
                  "requirement_id": requirement_id,
                  "severity": critique.severity.value,
                  "dimension": critique.dimension.value,
                  "is_placeholder": critique.error is not None,
                  "finding": (critique.finding or "")[:240],
                  "proposed_fix": (critique.proposed_fix or "")[:160],
                  "confidence": round(float(critique.confidence), 2),
                  "cited_evidence_count": len(critique.cited_evidence)})

        # Emit per-critique memory suggestions. Producers self-skip when
        # severity != BLOCKER, when keywords don't match, etc., so calling
        # all three on every critique is cheap and correct. ``converged``
        # is unknown at critic time — pass False; the finalize node
        # promotes the validation_status when the debate converges.
        audits = _audits_from(
            produce_constraint_from_blocker(
                critique, requirement_id=requirement_id,
                round_number=round_num, converged=False,
            ),
            produce_incident_from_blocker(
                critique, requirement_id=requirement_id,
                round_number=round_num, converged=False,
            ),
            produce_anti_pattern_from_blocker(
                critique, requirement_id=requirement_id,
                round_number=round_num, converged=False,
            ),
        )
        return {
            "critiques_by_round": {round_num: [critique.model_dump()]},
            "memory_audits": audits,
        }

    return critic_node


def _critic_sends(state: DebateState) -> list[Send]:
    """Build the parallel critic fan-out: one ``Send`` per critic role.

    The barrier reducer on ``critiques_by_round`` (add-append) merges
    their outputs before synthesize runs.
    """

    return [
        Send("critic", {**state, "_critic_role": role})
        for role in CRITIC_ROLES
    ]


def _dispatch_critics(state: DebateState) -> list[Send]:
    """Fan-out edge from generator. On generator failure, the graph
    routes straight to finalize (empty list → no Sends fire)."""

    if state.get("aborted_reason"):
        return []
    return _critic_sends(state)


def _make_synthesize_node(
    registry: RoleRegistry,
    emit: Callable[[dict], None] | None = None,
    disposition_sink: Callable[[list[dict]], None] | None = None,
) -> Callable[[DebateState], dict]:
    """Synthesize node — folds the round's critiques into a revised draft.

    Reads the latest draft from ``draft_history`` and this round's
    critiques from ``critiques_by_round[round]``, then calls
    ``JudgeRole.defend``. Increments ``round_number`` and appends the
    revised draft so the terminal score runs against it.
    """

    def synthesize_node(state: DebateState) -> dict:
        history = state.get("draft_history", [])
        if not history:
            return {
                "errors": [{"node": "synthesize", "round": 0,
                            "message": "no draft from generator"}],
                "aborted_reason": "synthesize called without a draft",
            }

        latest_dict = history[-1]["refined"]
        draft = Draft.model_validate(latest_dict)
        round_num = state.get("round_number", 0) + 1
        requirement_id = state["requirement"].get("id", "?")

        critiques_raw = state.get("critiques_by_round", {}).get(round_num, [])
        critiques = [Critique.model_validate(c) for c in critiques_raw]
        blocker_count, warning_count = _count_severities(critiques)

        if emit:
            emit({"phase": "refining", "debate_event": "synthesize_started",
                  "round": round_num, "requirement_id": requirement_id,
                  "critique_count": len(critiques),
                  "blocker_count": blocker_count,
                  "warning_count": warning_count,
                  "critic_roles": [c.author_role for c in critiques]})

        judge = registry.get("judge")
        context = RoleContext(
            role="judge",
            requirement_id=requirement_id,
            round_number=round_num,
            evidence=state.get("evidence_bag", {}),
        )

        try:
            rebuttal = judge.defend(draft, critiques, context)
        except Exception as exc:
            log.warning("refinery_synthesize_failed", error=str(exc))
            return {
                "errors": [{"node": "synthesize", "round": round_num,
                            "message": str(exc)}],
                "round_number": round_num,  # still advance so termination fires
            }

        record: DraftRecord = {
            "round": round_num,
            "refined": rebuttal.revised_draft.model_dump(),
            "author_role": "judge",
            "model_tier": state.get("base_model", ""),
        }
        if emit:
            counts = {
                "accepted": sum(1 for e in rebuttal.entries if e.action == "accepted"),
                "rejected": sum(1 for e in rebuttal.entries if e.action == "rejected"),
                "deferred": sum(1 for e in rebuttal.entries if e.action == "deferred"),
                "partial": sum(1 for e in rebuttal.entries if e.action == "partial"),
            }
            entries_preview = []
            for i, entry in enumerate(rebuttal.entries[:4]):
                crit = critiques[i] if i < len(critiques) else None
                entries_preview.append({
                    "critique_ref": entry.critique_ref,
                    "role": crit.author_role if crit else "",
                    "severity": crit.severity.value if crit else "",
                    "dimension": crit.dimension.value if crit else "",
                    "action": entry.action,
                    "rationale": (entry.rationale or "")[:160],
                })
            revised = rebuttal.revised_draft
            conv_status = revised.convergence_status
            emit({"phase": "refining", "debate_event": "synthesis_ready",
                  "round": round_num, "requirement_id": requirement_id,
                  "counts": counts,
                  "entries_preview": entries_preview,
                  "tradeoffs_count": len(revised.explicit_tradeoffs),
                  "tradeoffs_preview": [t[:140] for t in revised.explicit_tradeoffs[:3]],
                  "unresolved_count": len(revised.unresolved_points),
                  "open_questions_count": len(revised.open_questions),
                  "convergence_status": (
                      conv_status.value if hasattr(conv_status, "value")
                      else (conv_status or "")
                  )})

        # Disposition sink — feeds the role-weighting aggregator. Best-
        # effort: a sink failure must not break the debate. Map each
        # rebuttal entry back to its critique by ``c-{i}`` index so the
        # role/severity/dimension on the disposition row is authoritative.
        if disposition_sink is not None:
            refinery_run_id = state.get("refinery_run_id") or ""
            critique_by_ref = {f"c-{i}": c for i, c in enumerate(critiques)}
            rows: list[dict] = []
            for entry in rebuttal.entries:
                crit = critique_by_ref.get(entry.critique_ref)
                if crit is None:
                    continue
                rows.append({
                    "refinery_run_id": refinery_run_id,
                    "requirement_id": requirement_id,
                    "round_number": round_num,
                    "role": crit.author_role,
                    "severity": crit.severity.value,
                    "dimension": crit.dimension.value,
                    "action": entry.action,
                })
            if rows:
                try:
                    disposition_sink(rows)
                except Exception as exc:  # pragma: no cover — defensive
                    log.warning(
                        "refinery_disposition_sink_failed",
                        error=str(exc), count=len(rows),
                    )

        # Emit DECISION memory suggestions for rebuttal entries with
        # rationale that generalises (the producer's heuristic). At
        # synthesize time the debate hasn't terminated yet, so leave
        # validation_status open — finalize promotes it when converged.
        audits = _audits_from(
            produce_decision_from_rebuttal(
                rebuttal, critiques=critiques,
                requirement_id=requirement_id, round_number=round_num,
                converged=False,
            ),
        )
        return {
            "draft_history": [record],
            "rebuttals_by_round": {round_num: rebuttal.model_dump()},
            "round_number": round_num,
            "memory_audits": audits,
        }

    return synthesize_node


def _make_score_node(
    registry: RoleRegistry,
    emit: Callable[[dict], None] | None = None,
) -> Callable[[DebateState], dict]:
    """Score node — delegates to ``JudgeRole.score``."""

    def score_node(state: DebateState) -> dict:
        history = state.get("draft_history", [])
        if not history:
            return {
                "errors": [{"node": "score", "round": 0,
                            "message": "no draft to score"}],
            }

        round_num = state.get("round_number", 0)
        draft = Draft.model_validate(history[-1]["refined"])
        requirement_id = state["requirement"].get("id", "?")
        if emit:
            emit({"phase": "refining", "debate_event": "score_started",
                  "round": round_num, "requirement_id": requirement_id,
                  "target_iteration": draft.iteration})

        judge = registry.get("judge")
        context = RoleContext(
            role="judge",
            requirement_id=requirement_id,
            round_number=round_num,
            evidence=state.get("evidence_bag", {}),
        )
        try:
            score = judge.score(draft, context, trace=None)
        except Exception as exc:
            log.warning("refinery_score_failed", error=str(exc))
            return {
                "errors": [{"node": "score", "round": round_num,
                            "message": str(exc)}],
                # Stamp a 0-score so the router terminates on the next visit
                # (conditional edge after score) or finalize picks the best draft.
                "scores_by_round": {round_num: {
                    "dimensions": {}, "overall": 0.0, "passed": False,
                    "error": str(exc),
                }},
            }
        if emit:
            dims = {k: round(float(v), 3) for k, v in (score.dimensions or {}).items()}
            failing = [
                name for name, val in (score.dimensions or {}).items()
                if val < (score.thresholds or {}).get(name, score.overall_threshold)
            ]
            reasons_preview = {
                name: ((reason or "")[:160])
                for name, reason in (score.reasons or {}).items()
            }
            emit({"phase": "refining", "debate_event": "score_ready",
                  "round": round_num, "requirement_id": requirement_id,
                  "overall": round(float(score.overall), 3),
                  "passed": score.passed,
                  "overall_threshold": score.overall_threshold,
                  "dimensions": dims,
                  "failing_dimensions": failing,
                  "reasons": reasons_preview,
                  "disagreement_score": round(float(score.disagreement_score), 3),
                  "missing_external_info": score.missing_external_info,
                  "rule_violations_count": len(score.rule_violations),
                  "rule_warnings_count": len(score.rule_warnings),
                  "fallback_used": score.fallback_used,
                  "llm_short_circuited": score.llm_short_circuited})
        return {"scores_by_round": {round_num: score.model_dump()}}

    return score_node


def _make_research_node(
    registry: RoleRegistry,
    emit: Callable[[dict], None] | None = None,
    research_agent_factory: Callable[[], "ResearchAgent"] | None = None,
) -> Callable[[DebateState], dict]:
    """Research node.

    Invokes the layered-sourcing ResearchAgent when the Judge flags
    ``missing_external_info``. The agent's ``ResearchNote`` is appended
    to ``state['research_notes']``; the next synthesize round reads it
    out of the evidence bag.
    """

    def research_node(state: DebateState) -> dict:
        round_num = state.get("round_number", 0)
        requirement = state["requirement"]
        if emit:
            emit({"phase": "refining", "debate_event": "research_started",
                  "round": round_num,
                  "requirement_id": requirement.get("id", "?")})

        try:
            agent = (research_agent_factory or _default_research_agent)()
        except Exception as exc:
            log.warning("refinery_research_agent_build_failed", error=str(exc))
            return {
                "research_calls_used": state.get("research_calls_used", 0) + 1,
                "errors": [{"node": "research", "round": round_num,
                            "message": f"agent factory failed: {exc}"}],
            }

        try:
            note = agent.run(
                query=requirement.get("description") or requirement.get("title", ""),
                round_number=round_num,
            )
        except Exception as exc:
            log.warning("refinery_research_run_failed", error=str(exc))
            return {
                "research_calls_used": state.get("research_calls_used", 0) + 1,
                "errors": [{"node": "research", "round": round_num,
                            "message": str(exc)}],
            }

        if emit:
            emit({"phase": "refining", "debate_event": "research_ready",
                  "round": round_num,
                  "requirement_id": requirement.get("id", "?"),
                  "insights": len(note.validated_insights),
                  "tier_mix": dict(note.tier_mix_summary)})
        return {
            "research_notes": [note.model_dump()],
            "research_calls_used": state.get("research_calls_used", 0) + 1,
        }

    return research_node


def _make_escalate_node(
    emit: Callable[[dict], None] | None = None,
) -> Callable[[DebateState], dict]:
    """Pure state mutation — no LLM call. Increments the escalation
    level so ``_make_critic_node`` and ``_make_synthesize_node`` pick
    up the stronger model tier."""

    def escalate_node(state: DebateState) -> dict:
        new_level = state.get("escalation_level", 0) + 1
        if emit:
            emit({"phase": "refining", "debate_event": "escalation",
                  "new_tier": "strong", "escalation_level": new_level})
        # Switch the base model to the strong model so subsequent role
        # calls use it. base_model is an immutable input normally but
        # Escalation mutates it here as the single knob roles read.
        return {
            "escalation_level": new_level,
            "base_model": state.get("strong_model") or state.get("base_model", ""),
        }

    return escalate_node


def _make_reconcile_node(
    registry: RoleRegistry,
    emit: Callable[[dict], None] | None = None,
) -> Callable[[DebateState], dict]:
    """Short-circuit synthesis when max_rounds hit without convergence.

    Calls ``Judge.reconcile_unresolved`` with ALL critiques + rebuttals
    + scores accumulated so far, producing a Draft that documents what
    couldn't be resolved rather than forcing a synthesis that doesn't
    exist. A CONFLICT memory is emitted unconditionally by the memory
    producers when this node runs."""

    def reconcile_node(state: DebateState) -> dict:
        history = state.get("draft_history", [])
        if not history:
            return {
                "aborted_reason": "reconcile called without a draft",
            }
        round_num = state.get("round_number", 0)
        requirement_id = state["requirement"].get("id", "?")
        latest = Draft.model_validate(history[-1]["refined"])

        all_blockers = 0
        for _rn, crits in state.get("critiques_by_round", {}).items():
            all_blockers += _count_severities(crits)[0]
        if emit:
            emit({"phase": "refining", "debate_event": "reconcile_started",
                  "round": round_num, "requirement_id": requirement_id,
                  "rounds_executed": round_num,
                  "total_blockers": all_blockers})

        from dark_factory.api.refinery.contracts import EvaluationScore, Rebuttal

        # Collect all critiques + rebuttals + scores across rounds.
        all_critiques_raw: list[dict] = []
        for _rn, crits in sorted(state.get("critiques_by_round", {}).items()):
            all_critiques_raw.extend(crits)
        all_critiques = [Critique.model_validate(c) for c in all_critiques_raw]
        all_rebuttals = [
            Rebuttal.model_validate(r)
            for r in state.get("rebuttals_by_round", {}).values()
        ]
        all_scores = [
            EvaluationScore.model_validate(s)
            for s in state.get("scores_by_round", {}).values()
        ]

        judge = registry.get("judge").configure(
            model=state.get("base_model"),
            reasoning_effort=state.get("reasoning_effort"),
        )
        context = RoleContext(
            role="judge", requirement_id=requirement_id, round_number=round_num,
        )
        try:
            rebuttal = judge.reconcile_unresolved(
                latest, all_critiques, all_rebuttals, all_scores, context,
            )
        except Exception as exc:
            log.warning("refinery_reconcile_failed", error=str(exc))
            return {
                "errors": [{"node": "reconcile", "round": round_num,
                            "message": str(exc)}],
                "aborted_reason": f"reconcile failed: {exc}",
            }

        record: DraftRecord = {
            "round": round_num + 1,
            "refined": rebuttal.revised_draft.model_dump(),
            "author_role": "judge",
            "model_tier": state.get("base_model", ""),
        }
        if emit:
            revised = rebuttal.revised_draft
            emit({"phase": "refining", "debate_event": "reconcile_ready",
                  "round": round_num, "requirement_id": requirement_id,
                  "unresolved_count": len(revised.unresolved_points),
                  "unresolved_preview": [
                      p[:160] for p in revised.unresolved_points[:5]
                  ],
                  "open_questions_count": len(revised.open_questions),
                  "open_questions_preview": [
                      q[:160] for q in revised.open_questions[:5]
                  ],
                  "tradeoffs_count": len(revised.explicit_tradeoffs)})

        # Reconcile fires only on short-circuit (max_rounds without
        # convergence). Both producers below are scoped to that case:
        # the CONFLICT producer self-skips unless convergence_status is
        # SHORT_CIRCUITED (which Judge.reconcile_unresolved stamps), and
        # the HYPOTHESIS producer turns each open_question into a
        # verifiable hypothesis for future debates.
        rounds_executed = round_num + 1
        audits = _audits_from(
            produce_conflict_on_short_circuit(
                final_draft=rebuttal.revised_draft,
                critiques_by_round={
                    int(k): [Critique.model_validate(c) for c in v]
                    for k, v in state.get("critiques_by_round", {}).items()
                },
                requirement_id=requirement_id,
                rounds_executed=rounds_executed,
            ),
            produce_hypotheses_from_open_questions(
                final_draft=rebuttal.revised_draft,
                requirement_id=requirement_id,
                rounds_executed=rounds_executed,
            ),
        )
        return {
            "draft_history": [record],
            "rebuttals_by_round": {round_num + 1: rebuttal.model_dump()},
            "round_number": round_num + 1,
            "memory_audits": audits,
        }

    return reconcile_node


def _default_research_agent() -> "ResearchAgent":
    """Build a ResearchAgent from current settings + default providers.

    Degraded mode — no provider-trust learning is applied because this
    path has no metrics-client handle. The orchestrator
    (``stream.py``) always injects its own factory in production with
    ``load_adjusted_trust_weights`` already applied; this default only
    fires in standalone graph-runner / test contexts.
    """

    from dark_factory.api.refinery.research import (
        ResearchAgent,
        build_research_agent_from_config,
    )
    from dark_factory.config import PipelineConfig, load_settings

    try:
        cfg = load_settings().pipeline
    except Exception:
        cfg = PipelineConfig()

    log.info("debate_research_agent_default_no_learning")
    agent: ResearchAgent = build_research_agent_from_config(cfg)
    return agent


def _make_finalize_node(
    emit: Callable[[dict], None] | None = None,
    *,
    conflict_threshold: float = 0.4,
) -> Callable[[DebateState], dict]:
    """Finalize — pick the best-scoring draft and emit a minimal trace.

    Picks the best-scoring draft across rounds (ties broken by most
    recent), with an override for the short-circuit path: a reconcile-
    produced SHORT_CIRCUITED draft always wins regardless of score.
    """

    def finalize_node(state: DebateState) -> dict:
        history = state.get("draft_history", [])
        scores = state.get("scores_by_round", {})
        aborted = state.get("aborted_reason")

        if not history:
            # Generator crashed — emit an empty terminal state. Every
            # key a converged trace carries also appears here (empty),
            # so downstream consumers don't branch on convergence.
            return {
                "final_refined": None,
                "final_trace": {
                    "requirement_id": state["requirement"].get("id", "?"),
                    "refinery_run_id": state.get("refinery_run_id", ""),
                    "convergence_status": ConvergenceStatus.ABORTED.value,
                    "termination_reason": aborted or "no draft produced",
                    "rounds_executed": 0,
                    "draft_history": [],
                    "critiques_by_round": state.get("critiques_by_round", {}),
                    "rebuttals_by_round": state.get("rebuttals_by_round", {}),
                    "scores_by_round": state.get("scores_by_round", {}),
                    "final_score": None,
                    "research_notes": state.get("research_notes", []),
                    "escalation_level": state.get("escalation_level", 0),
                    "research_calls_used": state.get("research_calls_used", 0),
                    "errors": state.get("errors", []),
                    "memory_audits": list(state.get("memory_audits", [])),
                },
            }

        # When a reconcile node ran (short-circuit), its SHORT_CIRCUITED
        # draft is the authoritative terminal state — NOT the highest-
        # scored mid-debate draft. Operators need to see the documented-
        # unresolved version, not a shinier one that ignored last-round
        # blockers.
        last_draft = history[-1]["refined"]
        # Pick best-scored round regardless of path — finalize records
        # the best_round score even on short-circuit (it still went
        # through N rounds of scoring before reconcile).
        best_round = 0
        best_overall = -1.0
        for rn, s in scores.items():
            overall = s.get("overall", 0.0)
            if overall >= best_overall:
                best_overall = overall
                best_round = rn

        if last_draft.get("convergence_status") == ConvergenceStatus.SHORT_CIRCUITED.value:
            best_draft = last_draft
        else:
            best_draft = next(
                (d["refined"] for d in history if d["round"] == best_round),
                history[-1]["refined"],
            )

        # Convergence status: stamped by the node that produced the draft.
        # Judge.defend sets CONVERGED on the synthesis; reconcile sets
        # SHORT_CIRCUITED. Aborted is already handled above.
        convergence_status = best_draft.get(
            "convergence_status", ConvergenceStatus.CONVERGED.value,
        )

        termination_reason = "threshold_met"
        if aborted:
            termination_reason = "aborted"
            convergence_status = ConvergenceStatus.ABORTED.value
        elif convergence_status == ConvergenceStatus.SHORT_CIRCUITED.value:
            termination_reason = "max_rounds_without_convergence"

        # Finalize-time producers: CONFLICT on high mid-round
        # disagreement (separate from the short-circuit conflict
        # produced by reconcile_node) and HYPOTHESIS from any
        # open_questions on the converged draft. Both self-skip when
        # their preconditions don't hold.
        rounds_executed = state.get("round_number", 0)
        converged = convergence_status == ConvergenceStatus.CONVERGED.value
        critiques_typed = {
            int(k): [Critique.model_validate(c) for c in v]
            for k, v in state.get("critiques_by_round", {}).items()
        }
        rebuttals_typed: dict[int, Rebuttal] = {}
        if converged:
            rebuttals_typed = {
                int(k): Rebuttal.model_validate(v)
                for k, v in state.get("rebuttals_by_round", {}).items()
            }
        final_score = scores.get(best_round) or {}
        disagreement_score = float(final_score.get("disagreement_score", 0.0))
        finalize_audits = _audits_from(
            produce_conflict_on_disagreement(
                disagreement_score=disagreement_score,
                threshold=conflict_threshold,
                critiques_by_round=critiques_typed,
                rebuttals_by_round=rebuttals_typed,
                requirement_id=state["requirement"].get("id", "?"),
                rounds_executed=rounds_executed,
                converged=converged,
            ) if converged else MemoryProducerResult(
                skipped_reason="not converged",
            ),
            produce_hypotheses_from_open_questions(
                final_draft=Draft.model_validate(best_draft),
                requirement_id=state["requirement"].get("id", "?"),
                rounds_executed=rounds_executed,
            ),
        )

        # When the debate converged, promote prior in-flight
        # ``unvalidated`` audits to ``produced`` so the user-facing
        # Apply UI defaults to checked for them. Audits emitted by
        # reconcile (already ``validated``) and operator-dismissed ones
        # stay at their current status.
        accumulated_audits = list(state.get("memory_audits", []))
        if converged:
            from dark_factory.api.refinery.contracts import (
                ValidationStatus as _VS,
            )
            for audit in accumulated_audits:
                if audit.get("validation_status") == _VS.UNVALIDATED.value:
                    audit["validation_status"] = _VS.PRODUCED.value

        all_audits = accumulated_audits + finalize_audits

        trace = {
            "requirement_id": state["requirement"].get("id", "?"),
            "refinery_run_id": state.get("refinery_run_id", ""),
            "convergence_status": convergence_status,
            "termination_reason": termination_reason,
            "rounds_executed": rounds_executed,
            "draft_history": history,
            "critiques_by_round": state.get("critiques_by_round", {}),
            "rebuttals_by_round": state.get("rebuttals_by_round", {}),
            "scores_by_round": scores,
            "final_score": scores.get(best_round),
            "research_notes": state.get("research_notes", []),
            "escalation_level": state.get("escalation_level", 0),
            "research_calls_used": state.get("research_calls_used", 0),
            "errors": state.get("errors", []),
            "memory_audits": all_audits,
        }
        if emit:
            emit({"phase": "refining", "debate_event": "finalized",
                  "requirement_id": state["requirement"].get("id", "?"),
                  "convergence_status": convergence_status,
                  "rounds": rounds_executed,
                  "memory_audit_count": len(all_audits)})
        return {
            "final_refined": best_draft,
            "final_trace": trace,
            "memory_audits": finalize_audits,
        }

    return finalize_node


# ─────────────────────────────────────────────────────────────────────
# Public: build + invoke
# ─────────────────────────────────────────────────────────────────────


def build_debate_graph(
    config: PipelineConfig,
    *,
    registry: RoleRegistry | None = None,
    emit: Callable[[dict], None] | None = None,
    research_agent_factory: Callable[[], Any] | None = None,
    disposition_sink: Callable[[list[dict]], None] | None = None,
) -> Any:
    """Build the compiled LangGraph Runnable for a per-requirement debate.

    Topology:

        START → generator → (critic fan-out) → synthesize → score
                                                   ↓
        [router: finalize | research | escalate | reconcile | next_round]
            research → synthesize
            escalate → critic fan-out (strong model)
            reconcile → finalize (short-circuit)
            finalize → END

    Termination invariants: every loop back-edge increments a
    monotonically-growing counter (round_number, research_calls_used,
    escalation_level) that the router compares to a finite cap. Belt-
    and-braces: ``recursion_limit = max_rounds * 8 + 4``.
    """

    reg = registry or RoleRegistry(config)
    research_agent_factory_override = research_agent_factory
    graph: StateGraph = StateGraph(DebateState)
    graph.add_node("generator", _make_generator_node(reg, emit))
    graph.add_node("critic", _make_critic_node(reg, emit))
    graph.add_node(
        "synthesize",
        _make_synthesize_node(reg, emit, disposition_sink=disposition_sink),
    )
    graph.add_node("score", _make_score_node(reg, emit))
    graph.add_node(
        "research",
        _make_research_node(reg, emit, research_agent_factory_override),
    )
    graph.add_node("escalate", _make_escalate_node(emit))
    graph.add_node("reconcile", _make_reconcile_node(reg, emit))
    graph.add_node(
        "finalize",
        _make_finalize_node(
            emit,
            conflict_threshold=float(
                getattr(config, "refinery_conflict_emit_threshold", 0.4),
            ),
        ),
    )

    graph.add_edge(START, "generator")
    # On generator abort, skip the panel and go straight to finalize.
    # Otherwise, fan out to the four critics in parallel.
    # ``_route_after_generator`` returns either the string "finalize"
    # or a list of Send objects directly; the path map only needs the
    # string case.
    graph.add_conditional_edges(
        "generator",
        _route_after_generator,
        {"finalize": "finalize"},
    )
    graph.add_edge("critic", "synthesize")
    graph.add_edge("synthesize", "score")
    # Router: after score, branch on the unified gate output. The
    # ``next_round`` case returns a list of ``Send`` objects directly
    # (one per critic role) so LangGraph fans out to the panel in
    # parallel — same shape ``_dispatch_critics`` returns from the
    # generator edge. Returning a plain string would dispatch to the
    # single ``critic`` node with no ``_critic_role`` stamped and the
    # round would silently degrade to one placeholder critique.
    graph.add_conditional_edges(
        "score",
        _route_after_score,
        {
            "finalize": "finalize",
            "research": "research",
            "escalate": "escalate",
            "reconcile": "reconcile",
        },
    )
    # Research feeds back into synthesize so the next round's Judge
    # sees the validated insights.
    graph.add_edge("research", "synthesize")
    # Escalation re-runs the critic panel with the strong model.
    graph.add_conditional_edges(
        "escalate",
        _route_after_escalate,
        {"finalize": "finalize"},
    )
    # Reconcile is a terminal synthesis — straight to finalize.
    graph.add_edge("reconcile", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def _route_after_score(state: DebateState) -> str | list[Send]:
    """The terminal router out of the ``score`` node.

    Reads:
    - ``aborted_reason``                    → finalize (hard abort)
    - ``scores_by_round[round].passed``     → finalize (converged)
    - ``round_number >= max_rounds``        → reconcile (short-circuit)
    - ``missing_external_info`` + research budget left → research
    - ``disagreement_score`` high + escalation budget left → escalate
    - otherwise                             → critic fan-out (next round)

    The non-terminal branch returns a list of ``Send`` objects (one per
    critic role) so LangGraph re-dispatches the full panel in parallel.
    Returning a plain string ``"critic"`` would hit the single critic
    node with no ``_critic_role`` set and the round would degenerate to
    one placeholder critique.

    Terminal outcomes guaranteed by monotonically-increasing counters
    with hard caps — see plan's "Termination invariants" list.
    """

    if state.get("aborted_reason"):
        return "finalize"

    round_num = state.get("round_number", 0)
    scores = state.get("scores_by_round", {})
    score = scores.get(round_num) or {}

    # Converged: the gate passed.
    if score.get("passed") is True:
        return "finalize"
    # Hard-failure on the scoring call itself: finalize with best-so-far.
    if score.get("error") is not None:
        return "finalize"

    # Short-circuit on max_rounds WITHOUT convergence.
    max_rounds = state.get("max_rounds", 3)
    if round_num >= max_rounds:
        return "reconcile"

    # Research invocation on Judge's missing_external_info flag.
    research_budget = state.get("research_call_cap", 1)
    research_used = state.get("research_calls_used", 0)
    if score.get("missing_external_info") and research_used < research_budget:
        return "research"

    # Model escalation on high disagreement.
    disagreement = float(score.get("disagreement_score", 0.0))
    escalation_cap = state.get("escalation_cap", 1)
    escalation_level = state.get("escalation_level", 0)
    if disagreement >= 0.7 and escalation_level < escalation_cap:
        return "escalate"
    elif disagreement >= 0.7:
        # High disagreement with escalation budget exhausted: the debate
        # continues at the base model and ``round_number >= max_rounds``
        # will eventually short-circuit. Surface the budget exhaustion so
        # operators can see the disagreement-driven escalation path was
        # unavailable rather than silently re-looping.
        log.info(
            "debate_escalation_budget_exhausted",
            requirement_id=state["requirement"].get("id", "?"),
            disagreement=round(disagreement, 3),
            escalation_level=escalation_level,
            escalation_cap=escalation_cap,
            round_number=round_num,
        )

    # Default: another critic round. Fan out to the panel in parallel
    # via Send objects so each role gets its own ``_critic_role`` overlay.
    return _critic_sends(state)


def _fan_out_or_finalize(state: DebateState) -> list[Send] | str:
    """Conditional edge: route straight to ``finalize`` when the
    generator aborted, otherwise emit the parallel critic fan-out.

    Used from two places: out of ``generator`` (pre-first-round) and
    out of ``escalate_model`` (subsequent rounds at the stronger tier).
    """

    if state.get("aborted_reason"):
        return "finalize"
    return _critic_sends(state)


# Backward-compat aliases for call sites that still reference the old
# names; both are the same function now.
_route_after_generator = _fan_out_or_finalize
_route_after_escalate = _fan_out_or_finalize


def run_debate(
    *,
    req: dict,
    all_requirements: list[dict],
    run_context: dict | None,
    debate_config: "DebateConfig | None" = None,
    config: PipelineConfig,
    on_progress: Callable[[dict], None] | None = None,
    refinery_run_id: str = "",
    registry: RoleRegistry | None = None,
    research_agent_factory: Callable[[], Any] | None = None,
    disposition_sink: Callable[[list[dict]], None] | None = None,
    # Legacy kwarg shim — callers that still pass individual fields get
    # them packed into a DebateConfig here. Flag off once call sites are
    # migrated.
    **legacy_kwargs: Any,
) -> dict:
    """Run one per-requirement debate end-to-end.

    Returns the terminal ``DebateState`` as a plain dict so the caller
    can extract ``final_refined`` and ``final_trace`` without importing
    the TypedDict shape.

    Prefer the ``debate_config`` argument (a ``DebateConfig`` instance);
    individual kwargs still work during migration via ``legacy_kwargs``.

    ``research_agent_factory`` (optional) — injectable for tests so the
    research node uses a stubbed agent instead of the default one that
    builds against live Qdrant / Neo4j / OpenAI.
    """

    from dark_factory.api.refinery.debate.config import DebateConfig

    if debate_config is None:
        # Build from legacy kwargs + pipeline_config so old call sites
        # don't have to change in one shot. The debate_config path is
        # preferred for new code.
        debate_config = _legacy_to_debate_config(
            config, legacy_kwargs,
        )

    graph = build_debate_graph(
        config,
        registry=registry,
        emit=on_progress,
        research_agent_factory=research_agent_factory,
        disposition_sink=disposition_sink,
    )

    evidence_bag: dict[str, Any] = {
        "all_requirements": all_requirements,
        "run_context": run_context,
        "tmpdir": debate_config.tmpdir,
        "max_turns": debate_config.max_turns,
        "timeout_seconds": debate_config.timeout_seconds,
        "on_progress": on_progress,
    }
    # Promote role-weighting + judge-calibration multipliers to the
    # top of the evidence bag so JudgeRole.defend / score can pick
    # them up without digging through run_context. The orchestrator
    # stamps these keys when Postgres has signal; absent → identity.
    if isinstance(run_context, dict) and run_context.get("role_weights"):
        evidence_bag["role_weights"] = run_context["role_weights"]
    if isinstance(run_context, dict) and run_context.get("judge_threshold_multiplier"):
        evidence_bag["judge_threshold_multiplier"] = (
            run_context["judge_threshold_multiplier"]
        )
    if debate_config.evidence_extras:
        evidence_bag.update(debate_config.evidence_extras)

    initial_state: DebateState = {  # type: ignore[typeddict-item]
        "requirement": req,
        "all_requirements": all_requirements,
        "run_context": run_context,
        "refinery_run_id": refinery_run_id,
        "max_rounds": debate_config.max_rounds,
        "score_threshold": debate_config.score_threshold,
        "research_call_cap": debate_config.research_call_cap,
        "escalation_cap": debate_config.escalation_cap,
        "base_model": debate_config.base_model,
        "strong_model": debate_config.strong_model,
        "reasoning_effort": debate_config.reasoning_effort,
        "evidence_bag": evidence_bag,
        "draft_history": [],
        "critiques_by_round": {},
        "rebuttals_by_round": {},
        "scores_by_round": {},
        "research_notes": [],
        "research_calls_used": 0,
        "escalation_level": 0,
        "round_number": 0,
        "errors": [],
        "aborted_reason": None,
    }
    # LangGraph's recursion_limit protects against infinite loops
    # even though the happy path has no loops. Pre-size for critic
    # fan-out + research + escalation so we don't need to bump this later.
    recursion_limit = debate_config.max_rounds * 8 + 4
    terminal = graph.invoke(initial_state, {"recursion_limit": recursion_limit})
    return dict(terminal)


def _legacy_to_debate_config(
    config: PipelineConfig,
    legacy_kwargs: dict[str, Any],
) -> "DebateConfig":
    """Pack legacy positional-style kwargs into a ``DebateConfig``.

    Keeps pre-refactor call sites working without a signature churn.
    Drop once every caller has migrated to passing ``debate_config=``
    explicitly.
    """

    from dark_factory.api.refinery.debate.config import DebateConfig

    max_rounds = legacy_kwargs.get(
        "max_rounds", config.refinery_debate_max_rounds,
    )
    return DebateConfig(
        max_rounds=max_rounds,
        score_threshold=legacy_kwargs.get(
            "score_threshold", config.refinery_debate_threshold,
        ),
        research_call_cap=legacy_kwargs.get(
            "research_call_cap", config.refinery_research_cap,
        ),
        escalation_cap=legacy_kwargs.get(
            "escalation_cap", config.refinery_escalation_cap,
        ),
        base_model=legacy_kwargs.get("base_model", ""),
        strong_model=legacy_kwargs.get("strong_model") or legacy_kwargs.get("base_model", ""),
        reasoning_effort=legacy_kwargs.get("reasoning_effort", ""),
        tmpdir=legacy_kwargs.get("tmpdir", "/tmp/refinery"),
        max_turns=legacy_kwargs.get("max_turns", max(5, max_rounds * 5)),
        timeout_seconds=legacy_kwargs.get("timeout_seconds", 900.0),
        evidence_extras=dict(legacy_kwargs.get("evidence_extras") or {}),
    )
