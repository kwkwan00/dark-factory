"""Per-requirement debate episodes — memory + documentation artifact.

An *episode* is a self-contained narrative of one adversarial-panel debate
on one requirement. It serves two purposes:

* **Memory.** Persisted alongside the existing swarm Episode store
  (Neo4j ``:Episode`` + Qdrant ``episodes`` collection) so future debates'
  context builder can recall how the panel resolved similar requirements.
* **Documentation.** Rendered as Markdown and stored under
  ``refinery/{result_id}/episodes/{requirement_id}.md`` so operators can
  export a human-readable record of each requirement's refinement journey.

The episode is built *deterministically* from the existing ``DebateTrace`` —
no LLM call. The trace already carries every round's critiques, rebuttals,
scores, escalations, research notes, and unresolved tensions; the renderer
just structures them into a readable narrative.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel, Field

from dark_factory.api.refinery.contracts import ConvergenceStatus, Severity

log = structlog.get_logger()


class EpisodeKeyEvent(BaseModel):
    """One turning-point in the debate narrative — distinct from the
    raw per-round event stream, which is captured separately in the
    trace. Key events are the ones an operator (or a future debate)
    would care about: the first blocker raised by each critic, the
    Judge's first synthesis verdict, an escalation, a research call,
    the final outcome."""

    order: int
    round_number: int
    actor: str            # role name, "judge", "research", "system"
    kind: str             # blocker | warning | rebuttal | score | escalation | research | terminal
    headline: str         # one-sentence summary
    detail: str = ""      # optional longer-form context


class DebateEpisode(BaseModel):
    """The autobiographical record of one per-requirement debate."""

    requirement_id: str
    refinery_run_id: str
    title: str

    outcome: str          # converged | short_circuited | aborted
    rounds_executed: int
    escalation_level: int = 0
    research_calls_used: int = 0

    # 200-word-ish prose narrative built deterministically from the trace.
    # We avoid an LLM call here so the episode is free and reproducible;
    # future enhancement: optionally polish with the Judge model.
    summary: str

    final_overall_score: float | None = None
    final_dimensions: dict[str, float] = Field(default_factory=dict)
    # Continuous convergence signal: 0 = no debate, 1 = converged, else
    # best_overall / overall_threshold clamped to (0, 0.99).
    convergence_score: float = 0.0

    key_events: list[EpisodeKeyEvent] = Field(default_factory=list)

    unresolved_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    explicit_tradeoffs: list[str] = Field(default_factory=list)

    participants: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0

    def summary_blob(self) -> str:
        """Concatenated text used for semantic indexing in Qdrant.

        Includes outcome + summary + key-event headlines so vector recall
        can surface this episode by topic OR by what happened in it
        (e.g. "previous debate where Security blocked on auth-token leak")."""

        ke_text = " · ".join(
            f"{e.actor}:{e.kind}:{e.headline}" for e in self.key_events[:15]
        )
        parts = [
            f"requirement={self.requirement_id} outcome={self.outcome} "
            f"rounds={self.rounds_executed} escalations={self.escalation_level}",
            self.summary,
            f"key_events: {ke_text}",
        ]
        if self.unresolved_points:
            parts.append("unresolved: " + " | ".join(self.unresolved_points[:5]))
        return "\n".join(parts)


def compute_convergence_score(trace: dict | None) -> float:
    """Map a terminal ``final_trace`` to a single 0..1 progress signal.

    * No trace / no rounds executed → 0.0 (carry-forward path or aborted
      before the first score).
    * Converged → 1.0.
    * Debated but not converged → best round's overall score divided by
      the threshold it had to clear, clamped to (0, 0.99) so 1.0 is
      reserved exclusively for converged debates.

    Operators read this directly off the requirement card to gauge how
    much more detail a requirement still needs.
    """

    if not isinstance(trace, dict):
        return 0.0
    if (trace.get("convergence_status") or "") == ConvergenceStatus.CONVERGED.value:
        return 1.0
    scores = trace.get("scores_by_round") or {}
    if not scores:
        return 0.0
    best_overall = 0.0
    threshold = 0.0
    for score in scores.values():
        if not isinstance(score, dict):
            continue
        overall = score.get("overall")
        if isinstance(overall, (int, float)) and overall > best_overall:
            best_overall = float(overall)
        thr = score.get("overall_threshold")
        if isinstance(thr, (int, float)) and thr > threshold:
            threshold = float(thr)
    if threshold <= 0:
        return 0.0
    return max(0.0, min(0.99, best_overall / threshold))


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _collect_participants(trace: dict) -> list[str]:
    seen: list[str] = []
    for crits in (trace.get("critiques_by_round") or {}).values():
        for c in crits or []:
            role = (c or {}).get("author_role")
            if role and role not in seen:
                seen.append(role)
    # Generator + judge always present in any non-empty trace.
    for fixed in ("product", "judge"):
        if fixed not in seen:
            seen.append(fixed)
    if trace.get("research_calls_used"):
        seen.append("research")
    return seen


def _build_key_events(trace: dict) -> list[EpisodeKeyEvent]:
    events: list[EpisodeKeyEvent] = []
    order = 1

    # Generator's draft is the implicit kickoff.
    history = trace.get("draft_history") or []
    if history:
        first = history[0]
        refined = (first or {}).get("refined") or {}
        events.append(EpisodeKeyEvent(
            order=order, round_number=0, actor="product", kind="draft",
            headline=f"Product proposed initial draft \"{_truncate(refined.get('title', ''), 80)}\"",
        ))
        order += 1

    # Per-round critiques + judge synthesis + score.
    critiques_by_round = trace.get("critiques_by_round") or {}
    rebuttals_by_round = trace.get("rebuttals_by_round") or {}
    scores_by_round = trace.get("scores_by_round") or {}

    def _as_int(k: Any) -> int:
        try:
            return int(k)
        except (TypeError, ValueError):
            return 0

    for rn in sorted({_as_int(k) for k in critiques_by_round.keys()} | {_as_int(k) for k in scores_by_round.keys()}):
        crits = critiques_by_round.get(rn) or critiques_by_round.get(str(rn)) or []
        # First BLOCKER per role (most informative; skip rubber-stamps).
        for c in crits:
            if not isinstance(c, dict):
                continue
            sev = (c.get("severity") or "").lower()
            if sev not in (Severity.BLOCKER.value, Severity.WARNING.value):
                continue
            role = c.get("author_role") or "critic"
            finding = _truncate(c.get("finding", ""), 200)
            kind = sev
            events.append(EpisodeKeyEvent(
                order=order, round_number=rn, actor=role, kind=kind,
                headline=f"{role.upper()} ({c.get('dimension', '?')}) — {finding}",
                detail=_truncate(c.get("proposed_fix", ""), 240),
            ))
            order += 1

        # Judge synthesis verdict (compact).
        reb = rebuttals_by_round.get(rn) or rebuttals_by_round.get(str(rn))
        if isinstance(reb, dict):
            entries = reb.get("entries") or []
            accept = sum(1 for e in entries if isinstance(e, dict) and e.get("action") == "accepted")
            reject = sum(1 for e in entries if isinstance(e, dict) and e.get("action") == "rejected")
            defer = sum(1 for e in entries if isinstance(e, dict) and e.get("action") == "deferred")
            mode = reb.get("mode") or "synthesize"
            events.append(EpisodeKeyEvent(
                order=order, round_number=rn, actor="judge",
                kind="reconcile" if mode == "reconcile_unresolved" else "rebuttal",
                headline=(
                    f"Judge {mode}: {accept} accepted, {reject} rejected"
                    + (f", {defer} deferred" if defer else "")
                ),
            ))
            order += 1

        # Score verdict.
        score = scores_by_round.get(rn) or scores_by_round.get(str(rn))
        if isinstance(score, dict):
            overall = score.get("overall")
            passed = score.get("passed")
            verdict = "passed" if passed else "failed"
            dims = score.get("dimensions") or {}
            failing = [
                k for k, v in dims.items()
                if isinstance(v, (int, float))
                and v < (score.get("thresholds") or {}).get(k, score.get("overall_threshold", 0.7))
            ]
            fail_text = f" — failing: {', '.join(failing)}" if failing else ""
            events.append(EpisodeKeyEvent(
                order=order, round_number=rn, actor="judge", kind="score",
                headline=(
                    f"Score {overall:.2f} {verdict}" if isinstance(overall, (int, float))
                    else f"Score {verdict}"
                ) + fail_text,
            ))
            order += 1

    # Escalation marker.
    esc = int(trace.get("escalation_level") or 0)
    if esc:
        events.append(EpisodeKeyEvent(
            order=order, round_number=trace.get("rounds_executed", 0),
            actor="system", kind="escalation",
            headline=f"Escalated to strong model (level {esc})",
        ))
        order += 1

    # Research call(s).
    rc = int(trace.get("research_calls_used") or 0)
    if rc:
        notes = trace.get("research_notes") or []
        first_note = notes[0] if notes else {}
        insights = len((first_note or {}).get("validated_insights") or [])
        events.append(EpisodeKeyEvent(
            order=order, round_number=trace.get("rounds_executed", 0),
            actor="research", kind="research",
            headline=f"Research invoked ({rc} call/s, {insights} validated insight/s)",
        ))
        order += 1

    # Terminal outcome.
    status = trace.get("convergence_status") or "converged"
    reason = trace.get("termination_reason") or ""
    events.append(EpisodeKeyEvent(
        order=order, round_number=trace.get("rounds_executed", 0),
        actor="system", kind="terminal",
        headline=f"Outcome: {status}" + (f" ({reason})" if reason else ""),
    ))
    return events


def _build_summary(trace: dict, final_draft: dict) -> str:
    """Deterministic ~150-word prose narrative."""

    status = trace.get("convergence_status") or "converged"
    rounds = trace.get("rounds_executed", 0)
    esc = trace.get("escalation_level", 0)
    rc = trace.get("research_calls_used", 0)
    title = _truncate(final_draft.get("title", ""), 100)

    blockers = warnings = 0
    for crits in (trace.get("critiques_by_round") or {}).values():
        for c in (crits or []):
            sev = (c or {}).get("severity", "").lower() if isinstance(c, dict) else ""
            if sev == Severity.BLOCKER.value:
                blockers += 1
            elif sev == Severity.WARNING.value:
                warnings += 1

    scores = trace.get("scores_by_round") or {}
    final_score = None
    for k in sorted((scores or {}).keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        val = scores[k]
        if isinstance(val, dict) and val.get("overall") is not None:
            final_score = val.get("overall")
    score_text = f"final overall {final_score:.2f}" if isinstance(final_score, (int, float)) else "no final score"

    if status == ConvergenceStatus.CONVERGED.value:
        verdict = (
            f"The panel converged on this requirement after {rounds} round(s) "
            f"with {score_text}."
        )
    elif status == ConvergenceStatus.SHORT_CIRCUITED.value:
        verdict = (
            f"The debate short-circuited after {rounds} round(s) without converging. "
            f"The Judge produced a reconcile-unresolved draft rather than forcing a synthesis."
        )
    else:
        verdict = (
            f"The debate aborted after {rounds} round(s): "
            f"{trace.get('termination_reason') or 'no draft produced'}."
        )

    panel_activity = (
        f"Critics raised {blockers} blocker(s) and {warnings} warning(s) across {rounds} round(s)."
    )

    extra = []
    if esc:
        extra.append(f"The router escalated to the strong model {esc} time(s).")
    if rc:
        extra.append(f"Research was invoked {rc} time(s).")
    unresolved = final_draft.get("unresolved_points") or []
    if unresolved:
        extra.append(
            f"{len(unresolved)} unresolved point(s) were documented: "
            + _truncate("; ".join(unresolved), 200)
        )
    tradeoffs = final_draft.get("explicit_tradeoffs") or []
    if tradeoffs:
        extra.append(
            f"{len(tradeoffs)} explicit tradeoff(s) recorded: "
            + _truncate("; ".join(tradeoffs), 200)
        )

    title_text = f"This episode covers \"{title}\". " if title else ""
    return title_text + verdict + " " + panel_activity + (" " + " ".join(extra) if extra else "")


def episode_from_trace(
    trace: dict,
    *,
    refinery_run_id: str = "",
    title: str = "",
    duration_seconds: float = 0.0,
) -> DebateEpisode | None:
    """Build a ``DebateEpisode`` from a ``final_trace`` dict.

    Returns ``None`` if the trace is empty / malformed (carry-forward path);
    callers should treat that as "no episode emitted for this requirement"
    rather than surfacing an empty card.
    """

    if not isinstance(trace, dict) or not trace.get("requirement_id"):
        return None

    history = trace.get("draft_history") or []
    final_draft: dict = {}
    if history:
        last = history[-1] or {}
        final_draft = last.get("refined") or {}

    key_events = _build_key_events(trace)
    summary = _build_summary(trace, final_draft)

    # Final score breakdown
    final_score_block = trace.get("final_score") or {}
    final_overall = final_score_block.get("overall") if isinstance(final_score_block, dict) else None
    final_dims = (
        final_score_block.get("dimensions")
        if isinstance(final_score_block, dict)
        else {}
    ) or {}
    # Normalize float coercion so downstream JSON doesn't carry Decimal etc.
    final_dims = {
        str(k): float(v) for k, v in final_dims.items()
        if isinstance(v, (int, float))
    }

    return DebateEpisode(
        requirement_id=trace.get("requirement_id", ""),
        refinery_run_id=refinery_run_id or trace.get("refinery_run_id", ""),
        title=title or final_draft.get("title", ""),
        outcome=trace.get("convergence_status") or ConvergenceStatus.CONVERGED.value,
        rounds_executed=int(trace.get("rounds_executed", 0)),
        escalation_level=int(trace.get("escalation_level", 0)),
        research_calls_used=int(trace.get("research_calls_used", 0)),
        summary=summary,
        final_overall_score=(
            float(final_overall) if isinstance(final_overall, (int, float)) else None
        ),
        final_dimensions=final_dims,
        key_events=key_events,
        convergence_score=compute_convergence_score(trace),
        unresolved_points=list(final_draft.get("unresolved_points") or []),
        open_questions=list(final_draft.get("open_questions") or []),
        explicit_tradeoffs=list(final_draft.get("explicit_tradeoffs") or []),
        participants=_collect_participants(trace),
        duration_seconds=float(duration_seconds),
    )


_OUTCOME_BADGE = {
    ConvergenceStatus.CONVERGED.value: "✅ Converged",
    ConvergenceStatus.SHORT_CIRCUITED.value: "⚠️ Short-circuited (max rounds without convergence)",
    ConvergenceStatus.ABORTED.value: "❌ Aborted",
}


def render_episode_markdown(episode: DebateEpisode) -> str:
    """Render the episode as a single Markdown document.

    Shape is operator-facing: an executive summary at top, then a key-event
    timeline, then sections for unresolved points / open questions /
    tradeoffs. Designed to read like a postmortem, not a log dump.
    """

    lines: list[str] = []
    title = episode.title or episode.requirement_id
    lines.append(f"# Debate Episode — {title}")
    lines.append("")
    lines.append(f"- **Requirement ID:** `{episode.requirement_id}`")
    if episode.refinery_run_id:
        lines.append(f"- **Refinery run:** `{episode.refinery_run_id}`")
    lines.append(f"- **Outcome:** {_OUTCOME_BADGE.get(episode.outcome, episode.outcome)}")
    lines.append(f"- **Convergence score:** {episode.convergence_score:.2f} / 1.00")
    lines.append(f"- **Rounds executed:** {episode.rounds_executed}")
    if episode.escalation_level:
        lines.append(f"- **Escalations:** {episode.escalation_level}")
    if episode.research_calls_used:
        lines.append(f"- **Research calls:** {episode.research_calls_used}")
    if episode.participants:
        lines.append(f"- **Panel participants:** {', '.join(episode.participants)}")
    if isinstance(episode.final_overall_score, (int, float)):
        lines.append(f"- **Final overall score:** {episode.final_overall_score:.2f}")
    if episode.duration_seconds:
        lines.append(f"- **Duration:** {episode.duration_seconds:.1f}s")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(episode.summary)
    lines.append("")

    if episode.final_dimensions:
        lines.append("## Final Dimension Scores")
        lines.append("")
        lines.append("| Dimension | Score |")
        lines.append("| --- | --- |")
        for name, val in episode.final_dimensions.items():
            lines.append(f"| {name} | {val:.2f} |")
        lines.append("")

    if episode.key_events:
        lines.append("## Key Events")
        lines.append("")
        last_round = -1
        for ev in episode.key_events:
            if ev.round_number != last_round:
                lines.append(f"### Round {ev.round_number}")
                lines.append("")
                last_round = ev.round_number
            lines.append(f"- **{ev.actor}** · _{ev.kind}_ — {ev.headline}")
            if ev.detail:
                lines.append(f"  - fix: {ev.detail}")
        lines.append("")

    if episode.unresolved_points:
        lines.append("## Unresolved Points")
        lines.append("")
        for p in episode.unresolved_points:
            lines.append(f"- {p}")
        lines.append("")

    if episode.open_questions:
        lines.append("## Open Questions")
        lines.append("")
        for q in episode.open_questions:
            lines.append(f"- {q}")
        lines.append("")

    if episode.explicit_tradeoffs:
        lines.append("## Explicit Tradeoffs")
        lines.append("")
        for t in episode.explicit_tradeoffs:
            lines.append(f"- {t}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


_OUTCOME_TO_SWARM = {
    ConvergenceStatus.CONVERGED.value: "success",
    ConvergenceStatus.SHORT_CIRCUITED.value: "partial",
    ConvergenceStatus.ABORTED.value: "failed",
}


def write_debate_episode_to_memory(
    episode: DebateEpisode,
    *,
    memory_repo: Any = None,
    vector_repo: Any = None,
) -> bool:
    """Project a ``DebateEpisode`` onto the swarm ``Episode`` schema and
    persist via the existing ``EpisodeWriter`` so it lands in Neo4j
    (``:Episode`` node) + Qdrant (``episodes`` collection) for future
    debates' recall path.

    Returns ``True`` on successful Neo4j write, ``False`` otherwise.
    Best-effort: any failure is logged and swallowed.

    The mapping carries refinery-specific fields where the swarm schema
    has compatible slots:

    * ``run_id`` ← refinery_run_id
    * ``feature`` ← requirement_id (the per-debate "feature")
    * ``outcome`` ← converged/short_circuited/aborted → success/partial/failed
    * ``turns_used`` ← rounds_executed
    * ``key_events`` ← projected onto swarm shape
    * ``final_eval_scores`` ← final_dimensions
    * ``agents_visited`` ← participants
    * ``tool_calls_summary`` ← carries escalation_level + research_calls_used
    """

    if memory_repo is None and vector_repo is None:
        return False

    try:
        from datetime import datetime, timedelta, timezone

        from dark_factory.memory.episodes import (
            Episode as SwarmEpisode,
            EpisodeKeyEvent as SwarmKeyEvent,
            EpisodeWriter,
        )

        # Translate key events into the swarm shape. The swarm event
        # type uses (order, agent, event, description) — our richer
        # (kind, headline, detail, round_number) collapses cleanly:
        # detail (when present) gets prepended to description.
        swarm_key_events = [
            SwarmKeyEvent(
                order=ke.order,
                agent=ke.actor,
                event=ke.kind,
                description=(
                    f"[round {ke.round_number}] {ke.headline}"
                    + (f" — {ke.detail}" if ke.detail else "")
                ),
            )
            for ke in episode.key_events
        ]

        # Episode id is content-addressed on (run_id, feature) so re-writes
        # of the same debate are idempotent.
        episode_id = (
            f"refinery-episode-{episode.refinery_run_id}-{episode.requirement_id}"
        )
        ended_at = datetime.now(tz=timezone.utc)
        started_at = ended_at - timedelta(seconds=max(0.0, episode.duration_seconds))

        # Pull embeddings off the vector_repo if available — matches
        # the orchestrator's pattern.
        embedder = None
        if vector_repo is not None:
            embedder = getattr(vector_repo, "_embeddings", None)

        swarm_ep = SwarmEpisode(
            id=episode_id,
            run_id=episode.refinery_run_id or "refinery",
            feature=episode.requirement_id,
            outcome=_OUTCOME_TO_SWARM.get(episode.outcome, "partial"),
            summary=episode.summary,
            key_events=swarm_key_events,
            turns_used=episode.rounds_executed,
            duration_seconds=episode.duration_seconds,
            spec_ids=[],
            final_eval_scores=dict(episode.final_dimensions),
            agents_visited=list(episode.participants),
            tool_calls_summary={
                "escalations": episode.escalation_level,
                "research_calls": episode.research_calls_used,
            },
            recalled_memory_ids=[],
            started_at=started_at,
            ended_at=ended_at,
        )
        writer = EpisodeWriter(
            memory_repo=memory_repo,
            vector_repo=vector_repo,
            embeddings=embedder,
        )
        return writer.write(swarm_ep)
    except Exception as exc:
        log.warning(
            "refinery_episode_memory_write_failed",
            requirement_id=episode.requirement_id,
            refinery_run_id=episode.refinery_run_id,
            error=str(exc),
        )
        return False


__all__ = [
    "DebateEpisode",
    "EpisodeKeyEvent",
    "compute_convergence_score",
    "episode_from_trace",
    "render_episode_markdown",
    "write_debate_episode_to_memory",
]
