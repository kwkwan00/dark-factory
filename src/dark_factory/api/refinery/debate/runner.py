"""Runner that routes one requirement through the LangGraph debate.

Returns ``(RefinedRequirement, memories, output)`` so ``stream.py``'s
thread-pool + progress machinery can stay agnostic of the debate
internals. The legacy single-agent path has been removed; this is the
only Phase-2 refinement path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import structlog

from dark_factory.api.refinery.contracts import (
    Draft,
    RawRequirement,
)
from dark_factory.api.refinery.models import RefinedRequirement, SuggestedMemory
from dark_factory.api.refinery.roles.registry import RoleRegistry
from dark_factory.config import PipelineConfig

log = structlog.get_logger()


def run_phase2_generator(
    *,
    req: dict,
    all_requirements: list[dict],
    run_context: dict | None,
    tmpdir: str,
    max_turns: int,
    timeout_seconds: float,
    model: str,
    reasoning_effort: str,
    on_progress: Callable[[dict], None] | None,
    config: PipelineConfig,
    refinery_run_id: str = "",
    research_agent_factory: Callable[[], object] | None = None,
    disposition_sink: Callable[[list[dict]], None] | None = None,
    max_rounds_override: int | None = None,
) -> tuple[RefinedRequirement | None, list[SuggestedMemory], str]:
    """Run the Phase-2+ debate surface.

    Routes the call through the LangGraph debate subgraph
    (``generator → critics → synthesize → score → finalize``). The
    output shape stays the legacy ``(RefinedRequirement, memories,
    output)`` tuple so ``stream.py`` is agnostic to which internal
    path ran.
    """

    req_id = req.get("id", "unknown")

    # Accumulator buffer so ProductRole can emit legacy memories through
    # the evidence bag without us threading a second return value.
    memories_buf: list[SuggestedMemory] = []

    registry = RoleRegistry(config)
    if not registry.has("product") or not registry.has("judge"):
        # The adversarial panel requires Product (generator) and Judge
        # (synthesis + scoring). If either is missing the registry is
        # misconfigured — surface the error rather than silently producing
        # a degraded result.
        raise RuntimeError(
            f"refinery debate cannot run without product + judge roles "
            f"(have_product={registry.has('product')}, "
            f"have_judge={registry.has('judge')})"
        )

    # Thread the memory accumulator through the evidence bag so the
    # generator's legacy-memory emission still reaches the SSE caller.
    from dark_factory.api.refinery.debate.graph import run_debate

    # Wrap the caller's on_progress to tag events with the requirement id.
    def _on_progress_tagged(event: dict) -> None:
        event.setdefault("requirement_id", req_id)
        if on_progress:
            on_progress(event)

    from dark_factory.api.refinery.debate.config import DebateConfig

    # Preserve the caller's max_turns + timeout so ProductRole.propose
    # forwards the legacy runner's LLM turn budget unchanged. The graph's
    # default (max_rounds*5) kicks in only when the caller doesn't specify.
    debate_config = DebateConfig.from_settings(
        config,
        base_model=model,
        reasoning_effort=reasoning_effort,
        tmpdir=tmpdir,
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
        evidence_extras={"suggested_memories_accumulator": memories_buf},
    )
    if max_rounds_override is not None and max_rounds_override != debate_config.max_rounds:
        debate_config = replace(debate_config, max_rounds=int(max_rounds_override))
    terminal_state = run_debate(
        req=_normalize_req(req),
        all_requirements=all_requirements,
        run_context=run_context,
        debate_config=debate_config,
        config=config,
        on_progress=_on_progress_tagged,
        refinery_run_id=refinery_run_id,
        registry=registry,
        research_agent_factory=research_agent_factory,
        disposition_sink=disposition_sink,
    )

    final_refined = terminal_state.get("final_refined")
    if final_refined is None:
        # Generator crashed or terminal state is malformed; produce a
        # carry-forward RefinedRequirement so downstream stream.py
        # machinery doesn't see None.
        log.warning("refinery_debate_no_final_refined", req_id=req_id)
        return (
            _carry_forward_refined(req),
            list(memories_buf),
            terminal_state.get("final_trace", {}).get("termination_reason", ""),
        )

    draft = Draft.model_validate(final_refined)
    raw = RawRequirement.model_validate(_normalize_req(req))
    refined = _draft_to_refined(draft, raw)
    final_trace = terminal_state.get("final_trace")
    refined.debate = _trace_to_debate_summary(final_trace)
    from dark_factory.api.refinery.episode import compute_convergence_score
    refined.convergence_score = compute_convergence_score(final_trace)

    # Attach the per-requirement Debate Episode to the wire payload so the
    # frontend tab can render it without a follow-up fetch. The full trace
    # remains in ``trace.json`` for forensic depth; the episode is the
    # operator-readable narrative built deterministically from it.
    try:
        from dark_factory.api.refinery.episode import (
            episode_from_trace,
            render_episode_markdown,
        )
        episode = episode_from_trace(
            terminal_state.get("final_trace") or {},
            refinery_run_id=refinery_run_id,
            title=draft.title,
        )
        if episode is not None and refined.debate is not None:
            refined.debate["episode"] = episode.model_dump()
            refined.debate["episode_markdown"] = render_episode_markdown(episode)
    except Exception as exc:  # pragma: no cover — best-effort
        log.warning("refinery_episode_build_failed", req_id=req_id, error=str(exc))

    return refined, list(memories_buf), ""


def _trace_to_debate_summary(trace: dict | None) -> dict | None:
    """Project the runner's ``final_trace`` dict onto the wire-format
    ``debate`` field on ``RefinedRequirement``. Returns ``None`` when no
    trace was produced (carry-forward path) so the frontend can hide the
    debate disclosure cleanly."""

    if not isinstance(trace, dict):
        return None

    # Strip critique/rebuttal payloads to the fields the UI needs — the
    # full nested trace is archived to ``trace.json`` separately and
    # doesn't need to ride the per-requirement response.
    def _critiques(rounds: dict | None) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for k, v in (rounds or {}).items():
            if not isinstance(v, list):
                continue
            out[str(k)] = [
                {
                    "role": c.get("author_role") if isinstance(c, dict) else "",
                    "severity": c.get("severity") if isinstance(c, dict) else "",
                    "dimension": c.get("dimension") if isinstance(c, dict) else "",
                    "finding": c.get("finding", "") if isinstance(c, dict) else "",
                    "proposed_fix": c.get("proposed_fix", "") if isinstance(c, dict) else "",
                }
                for c in v
            ]
        return out

    def _rebuttals(rounds: dict | None) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for k, v in (rounds or {}).items():
            if not isinstance(v, dict):
                continue
            entries = v.get("entries") or []
            out[str(k)] = {
                "accepted_count": sum(
                    1 for e in entries
                    if isinstance(e, dict) and e.get("action") == "accepted"
                ),
                "rejected_count": sum(
                    1 for e in entries
                    if isinstance(e, dict) and e.get("action") == "rejected"
                ),
                "mode": v.get("mode", ""),
            }
        return out

    def _scores(rounds: dict | None) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for k, v in (rounds or {}).items():
            if not isinstance(v, dict):
                continue
            out[str(k)] = {
                "overall": v.get("overall"),
                "passed": v.get("passed"),
                "dimensions": v.get("dimensions") or {},
            }
        return out

    return {
        "rounds_executed": trace.get("rounds_executed", 0),
        "convergence_status": trace.get("convergence_status"),
        "termination_reason": trace.get("termination_reason", ""),
        "escalation_level": trace.get("escalation_level", 0),
        "research_calls_used": trace.get("research_calls_used", 0),
        "critiques_by_round": _critiques(trace.get("critiques_by_round")),
        "rebuttals_by_round": _rebuttals(trace.get("rebuttals_by_round")),
        "scores_by_round": _scores(trace.get("scores_by_round")),
    }


def _normalize_req(req: dict) -> dict:
    """Coerce the raw ``req`` dict into a shape RawRequirement can parse."""

    return {
        "id": req.get("id", "unknown"),
        "title": req.get("title", "") or "",
        "description": req.get("description", "") or "",
        "priority": req.get("priority") or "medium",
        "tags": list(req.get("tags") or []),
        "source_file": req.get("source_file") or "",
    }


def _carry_forward_refined(req: dict) -> RefinedRequirement:
    return RefinedRequirement(
        id=req.get("id", "unknown"),
        original_title=req.get("title", "") or "",
        original_description=req.get("description", "") or "",
        title=req.get("title", "") or "",
        description=req.get("description", "") or "",
        priority=req.get("priority") or "medium",
        tags=list(req.get("tags") or []),
        changes=[],
        pass_context="Debate graph did not produce a final draft.",
    )


def _draft_to_refined(draft: Draft, raw: RawRequirement) -> RefinedRequirement:
    """Inverse of ``_refined_to_draft`` — maps a Draft onto the legacy
    RefinedRequirement shape that the rest of the refinery (storage,
    markdown report, frontend) already understands."""

    from dark_factory.api.refinery.models import RequirementRelationship, SuggestedSpec

    return RefinedRequirement(
        id=draft.requirement_id,
        original_title=raw.title,
        original_description=raw.description,
        title=draft.title,
        description=draft.description,
        priority=draft.priority,
        tags=list(draft.tags),
        relationships=[
            RequirementRelationship(
                target_id=r.target_id,
                type=r.type,
                rationale=r.rationale,
            )
            for r in draft.relationships
        ],
        suggested_specs=[
            SuggestedSpec(
                title=s.title,
                capability=s.capability,
                description=s.description,
                acceptance_criteria=list(s.acceptance_criteria),
            )
            for s in draft.suggested_specs
        ],
        # ``changes`` and ``pass_context`` are populated post-synthesis
        # from the Rebuttal ledger + Judge reasoning.
        changes=[],
        pass_context="",
        # Surface convergence metadata so the frontend can show NOT-
        # CONVERGED badges + documented-unresolved sections.
        convergence_status=(
            draft.convergence_status.value if draft.convergence_status else None
        ),
        unresolved_points=list(draft.unresolved_points),
        open_questions=list(draft.open_questions),
        explicit_tradeoffs=list(draft.explicit_tradeoffs),
    )
