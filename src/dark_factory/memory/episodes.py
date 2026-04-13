"""Episodic memory: autobiographical records of individual feature runs.

Each Episode is the narrative trajectory of one feature swarm running
to completion (success) or termination (failure). It complements the
four semantic memory types (Pattern, Mistake, Solution, Strategy) by
capturing **what actually happened on a specific run** rather than
**what generalises across runs**.

Why both? Semantic memory answers *"what should I do?"*; episodic
memory answers *"what worked last time I was in this exact
situation?"*. A Planner about to tackle a feature called ``auth``
benefits from both: the generalised Pattern memories tell it to use
parameterised queries for SQL, while the episodic memories tell it
that the last three runs of ``auth`` succeeded with JWT after
initially trying OAuth.

The system has three pieces:

1. **`synthesize_episode`** — takes a completed ``FeatureResult`` +
   progress-broker history and asks a small LLM call to produce a
   narrative summary (~200 words) plus a structured list of key
   turning-point events. Uses ``llm.complete_structured`` so we get
   validated pydantic output rather than loose prose parsing.

2. **`EpisodeWriter`** — embeds the summary via the existing
   ``EmbeddingService`` and writes to both Neo4j (structured graph
   with a PRODUCED_IN relationship to the Run node) and Qdrant (the
   ``dark_factory_episodes`` collection for semantic retrieval).
   Every step is best-effort: a Neo4j outage, an OpenAI embedding
   failure, or a Qdrant hiccup logs a warning but never propagates
   out of the writer. Losing an episode is acceptable; failing a
   pipeline run because we couldn't log an episode is not.

3. **Recall tool** (in ``agents/tools.py``) — ``recall_episodes``
   hybrid-searches episodes the same way ``recall_memories`` does:
   keyword match on Neo4j + vector match on Qdrant, merged via
   reciprocal rank fusion. Planner calls it at the start of every
   feature before picking a strategy.

Episodes are cheap to write and cheap to recall but the LLM
summarisation call does add ~1k tokens per feature, so the whole
subsystem is gated on ``settings.pipeline.enable_episodic_memory``
which defaults to ``True``. Turn it off if you are burning too much
budget on feature runs that never recur.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from dark_factory.agents.cancellation import PipelineCancelled

if TYPE_CHECKING:
    from dark_factory.llm.base import LLMClient
    from dark_factory.memory.repository import MemoryRepository
    from dark_factory.vector.embeddings import EmbeddingService
    from dark_factory.vector.repository import VectorRepository

log = structlog.get_logger()


# ── Public models ────────────────────────────────────────────────────────────


class EpisodeKeyEvent(BaseModel):
    """One turning-point event in an episode's narrative.

    The agent list (not the raw progress event stream) — think
    "decisions the Planner made", "reviewer rejections", "strategy
    pivots" rather than "tool call X succeeded".
    """

    order: int
    """1-indexed position in the narrative sequence."""

    agent: str
    """Which swarm role ('planner', 'coder', 'reviewer', 'tester',
    or 'system' for orchestrator-level events)."""

    event: str
    """Short label like 'strategy_picked', 'handoff', 'rejection',
    'approval', 'test_pass', 'error'."""

    description: str
    """One-sentence explanation of what happened at this step."""


class Episode(BaseModel):
    """The autobiographical record of one feature swarm execution."""

    id: str
    """Content-addressed episode id: stable hash over (run_id,
    feature). Idempotent re-writes of the same episode update the
    existing node rather than creating a duplicate."""

    run_id: str
    feature: str

    outcome: str  # "success" | "partial" | "failed"
    """The outcome of the feature swarm. ``partial`` means the swarm
    ran to completion but some tests or evaluations failed;
    ``failed`` means the swarm itself errored or was cancelled."""

    summary: str
    """LLM-generated ~200-word narrative of what happened. This is
    the text we embed for semantic recall."""

    key_events: list[EpisodeKeyEvent] = Field(default_factory=list)

    turns_used: int = 0
    """Number of agent handoffs the swarm burned."""

    duration_seconds: float = 0.0

    spec_ids: list[str] = Field(default_factory=list)

    final_eval_scores: dict[str, float] = Field(default_factory=dict)
    """Flattened {metric_name: average_score} across the feature's
    specs. Easier for the LLM to reason about than the nested
    per-spec shape."""

    agents_visited: list[str] = Field(default_factory=list)

    tool_calls_summary: dict[str, int] = Field(default_factory=dict)
    """Map of tool_name → call count, capped to the top 10 most-used
    tools so the Qdrant payload stays bounded."""

    started_at: datetime
    ended_at: datetime

    def summary_blob(self) -> str:
        """Build the text we actually embed for semantic retrieval.

        Includes the feature name + outcome + summary + a compact
        key-event list. Embedding just the summary alone loses the
        structural cues (outcome, feature name) that make hybrid
        recall work well, so we concatenate.
        """
        key_events_text = " · ".join(
            f"{ke.agent}:{ke.event}" for ke in self.key_events[:6]
        )
        return (
            f"feature={self.feature} outcome={self.outcome} "
            f"turns={self.turns_used} duration={self.duration_seconds:.0f}s\n"
            f"{self.summary}\n"
            f"key_events: {key_events_text}"
        )


# ── LLM summariser ──────────────────────────────────────────────────────────


class _EpisodeSynthesis(BaseModel):
    """Structured output of the summariser LLM call."""

    summary: str = Field(
        description=(
            "200-word prose narrative of what happened during this "
            "feature execution. Focus on the trajectory of decisions: "
            "what strategy was chosen, what blocked progress, how it "
            "was resolved, what the final outcome was. Do NOT repeat "
            "the raw event list — synthesise it into a story a "
            "future Planner can skim in 10 seconds."
        )
    )
    key_events: list[EpisodeKeyEvent] = Field(
        default_factory=list,
        description=(
            "3-8 turning-point events in order. Skip routine handoffs "
            "and routine tool calls; include only the events a future "
            "Planner would care about (strategy picks, rejections, "
            "pivots, test passes/failures)."
        ),
    )


from dark_factory.prompts import get_prompt

_SYNTHESIS_SYSTEM_PROMPT = get_prompt("episode_synthesis", "system")


def _build_synthesis_prompt(
    *,
    feature: str,
    spec_ids: list[str],
    outcome: str,
    turns_used: int,
    duration_seconds: float,
    final_eval_scores: dict[str, float],
    agents_visited: list[str],
    tool_calls_summary: dict[str, int],
    progress_events: list[dict[str, Any]],
    error: str | None = None,
) -> str:
    # Cap the progress event list so a pathologically long run can't
    # blow up the prompt. The summariser only needs the salient slice;
    # reviewer agents can always re-query Postgres for the full
    # forensic log.
    capped = progress_events[-80:] if len(progress_events) > 80 else progress_events
    events_text = "\n".join(
        f"  - {e.get('event', '?')}: "
        f"{', '.join(f'{k}={v}' for k, v in e.items() if k not in ('event', 'timestamp'))[:200]}"
        for e in capped
    )
    tools_text = ", ".join(
        f"{name}×{count}"
        for name, count in sorted(
            tool_calls_summary.items(), key=lambda x: -x[1]
        )[:10]
    ) or "(none)"
    scores_text = ", ".join(
        f"{k}={v:.2f}" for k, v in final_eval_scores.items()
    ) or "(none)"

    error_block = f"\n\nError: {error}" if error else ""

    return f"""\
Feature: {feature}
Outcome: {outcome}
Turns used: {turns_used}
Duration: {duration_seconds:.1f}s
Specs attempted: {', '.join(spec_ids) or '(none)'}
Agents visited: {', '.join(agents_visited) or '(none)'}
Tool calls (top 10): {tools_text}
Final eval scores: {scores_text}{error_block}

Progress events (latest {len(capped)} of {len(progress_events)}):
{events_text or '  (no progress events captured)'}

Write the summary and key_events now."""


def _fallback_summary(
    feature: str,
    outcome: str,
    turns_used: int,
    duration_seconds: float,
    error: str | None,
) -> str:
    """Cheap deterministic fallback when the LLM summariser fails.

    We still want an episode row so the UI can show "run X attempted
    feature Y" even if we couldn't afford the narrative generation.
    """
    base = (
        f"Feature '{feature}' finished with outcome '{outcome}' "
        f"after {turns_used} turns ({duration_seconds:.1f}s)."
    )
    if error:
        base += f" Error: {error[:200]}"
    base += " (LLM summariser unavailable — deterministic fallback.)"
    return base


def synthesize_episode(
    *,
    run_id: str,
    feature: str,
    spec_ids: list[str],
    outcome: str,
    turns_used: int,
    duration_seconds: float,
    started_at: datetime,
    ended_at: datetime,
    final_eval_scores: dict[str, float],
    agents_visited: list[str],
    tool_calls_summary: dict[str, int],
    progress_events: list[dict[str, Any]],
    llm: "LLMClient | None",
    error: str | None = None,
) -> Episode:
    """Assemble an Episode from a completed feature's in-memory state.

    Calls the LLM summariser if one is available, otherwise falls back
    to a deterministic template so every completed feature still
    produces an episode row. Never raises — failures are logged and
    translated into the fallback path.
    """
    key_events: list[EpisodeKeyEvent] = []
    summary = ""

    if llm is not None:
        prompt = _build_synthesis_prompt(
            feature=feature,
            spec_ids=spec_ids,
            outcome=outcome,
            turns_used=turns_used,
            duration_seconds=duration_seconds,
            final_eval_scores=final_eval_scores,
            agents_visited=agents_visited,
            tool_calls_summary=tool_calls_summary,
            progress_events=progress_events,
            error=error,
        )
        try:
            synthesis = llm.complete_structured(
                prompt=prompt,
                response_model=_EpisodeSynthesis,
                system=_SYNTHESIS_SYSTEM_PROMPT,
            )
            summary = (synthesis.summary or "").strip()
            key_events = list(synthesis.key_events)
        except PipelineCancelled:
            # B2 fix: cooperative cancel signal must propagate out
            # of the episode synthesiser. Without this guard, the
            # broad ``except Exception`` would swallow the cancel
            # and episode writing would fall through to the
            # deterministic template, leaving the pipeline to keep
            # running even though the user clicked Cancel.
            raise
        except Exception as exc:
            log.warning(
                "episode_synthesis_llm_failed",
                feature=feature,
                run_id=run_id,
                error=str(exc),
            )

    if not summary:
        summary = _fallback_summary(
            feature=feature,
            outcome=outcome,
            turns_used=turns_used,
            duration_seconds=duration_seconds,
            error=error,
        )
        # L7 fix: when we fall back to the deterministic template
        # (LLM unavailable or the call raised), record an incident
        # row so the Run Detail popup shows "episode X used fallback
        # template". Without this, episode fallback is silent — the
        # episode still appears in the UI but the operator has no
        # way to tell it lacks the narrative richness of an
        # LLM-synthesised summary. Best-effort: any failure in the
        # incident path is swallowed so a metrics outage can't break
        # episode writing.
        try:
            from dark_factory.metrics.helpers import record_incident

            record_incident(
                category="memory",
                severity="warning",
                message=(
                    f"episode synthesis fell back to deterministic template "
                    f"for feature '{feature}'"
                ),
                phase="episode_synthesis",
                feature=feature,
                run_id=run_id,
            )
        except Exception:  # pragma: no cover — defensive
            pass

    # Content-addressed id: idempotent re-writes replace the same
    # node rather than accumulating duplicates.
    id_material = f"{run_id}\n{feature}".encode()
    episode_id = (
        "ep-" + hashlib.sha256(id_material).hexdigest()[:16]
    )

    # Cap the tool-calls dict to the top 10 so the Qdrant payload
    # doesn't balloon when a feature makes 500 distinct tool calls.
    capped_tools = dict(
        sorted(tool_calls_summary.items(), key=lambda x: -x[1])[:10]
    )

    return Episode(
        id=episode_id,
        run_id=run_id,
        feature=feature,
        outcome=outcome,
        summary=summary,
        key_events=key_events,
        turns_used=turns_used,
        duration_seconds=duration_seconds,
        spec_ids=spec_ids,
        final_eval_scores=final_eval_scores,
        agents_visited=agents_visited,
        tool_calls_summary=capped_tools,
        started_at=started_at,
        ended_at=ended_at,
    )


# ── Writer ──────────────────────────────────────────────────────────────────


class EpisodeWriter:
    """Persist episodes to Neo4j + Qdrant, best-effort.

    Holds references to the memory repository, the vector repository,
    and the embedding service. Any of these can be ``None`` — the
    writer's job is to do what it can and log warnings for the rest.
    """

    def __init__(
        self,
        *,
        memory_repo: "MemoryRepository | None",
        vector_repo: "VectorRepository | None",
        embeddings: "EmbeddingService | None",
    ) -> None:
        self.memory_repo = memory_repo
        self.vector_repo = vector_repo
        self.embeddings = embeddings

    def write(self, episode: Episode) -> bool:
        """Write the episode to both stores.

        Returns ``True`` if at least the Neo4j write succeeded;
        ``False`` if the episode was lost entirely. Never raises.
        """
        neo4j_ok = False
        if self.memory_repo is not None:
            try:
                self.memory_repo.write_episode(episode)
                neo4j_ok = True
            except Exception as exc:
                log.warning(
                    "episode_neo4j_write_failed",
                    episode_id=episode.id,
                    feature=episode.feature,
                    run_id=episode.run_id,
                    error=str(exc),
                )
        else:
            log.debug(
                "episode_skipped_no_memory_repo",
                episode_id=episode.id,
            )

        # Embedding + Qdrant upsert is secondary — a successful Neo4j
        # write is enough for the UI to show the episode; the vector
        # side is only needed for semantic recall in future runs.
        if self.vector_repo is not None and self.embeddings is not None:
            try:
                [vector] = self.embeddings.embed_batch(
                    [episode.summary_blob()]
                )
                self.vector_repo.upsert_episode(
                    episode_id=episode.id,
                    run_id=episode.run_id,
                    feature=episode.feature,
                    outcome=episode.outcome,
                    summary=episode.summary,
                    vector=vector,
                    turns_used=episode.turns_used,
                    duration_seconds=episode.duration_seconds,
                )
            except Exception as exc:
                log.warning(
                    "episode_vector_upsert_failed",
                    episode_id=episode.id,
                    feature=episode.feature,
                    error=str(exc),
                )

        return neo4j_ok


# ── Helpers for consumers ────────────────────────────────────────────────────


def episode_from_feature_result(
    *,
    run_id: str,
    feature_result: dict[str, Any],
    started_at: datetime | None,
    progress_events: list[dict[str, Any]],
    llm: "LLMClient | None",
) -> Episode:
    """Shorthand: build an Episode from the orchestrator's
    ``FeatureResult`` shape.

    The orchestrator stores feature results as dicts with keys
    ``feature``, ``spec_ids``, ``status``, ``stats``, ``eval_scores``,
    ``error``. This helper unpacks them, computes a flattened eval
    score dict, and calls :func:`synthesize_episode`.
    """
    feature = feature_result.get("feature", "?")
    spec_ids = list(feature_result.get("spec_ids") or [])
    raw_status = feature_result.get("status", "unknown")
    outcome = _normalise_outcome(raw_status, feature_result.get("eval_scores"))
    stats = feature_result.get("stats") or {}
    turns_used = int(stats.get("agent_transitions", 0) or 0)
    duration = float(stats.get("duration_seconds", 0.0) or 0.0)
    agents_visited = list(stats.get("unique_agents_visited") or [])
    tool_calls = _extract_tool_call_counts(stats)

    final_scores = _flatten_eval_scores(feature_result.get("eval_scores") or {})

    now = datetime.now(timezone.utc)
    start = started_at or (
        now
        if duration <= 0
        else datetime.fromtimestamp(now.timestamp() - duration, tz=timezone.utc)
    )

    return synthesize_episode(
        run_id=run_id,
        feature=feature,
        spec_ids=spec_ids,
        outcome=outcome,
        turns_used=turns_used,
        duration_seconds=duration,
        started_at=start,
        ended_at=now,
        final_eval_scores=final_scores,
        agents_visited=agents_visited,
        tool_calls_summary=tool_calls,
        progress_events=progress_events,
        llm=llm,
        error=feature_result.get("error"),
    )


def _normalise_outcome(raw_status: str, eval_scores: Any) -> str:
    """Map the orchestrator's status strings to the episode outcome.

    ``success`` with failing evals becomes ``partial`` so the
    planner can distinguish "it ran but wasn't great" from "it ran
    cleanly". ``error`` / ``skipped`` both collapse to ``failed``.
    """
    status = (raw_status or "").lower()
    if status == "success":
        if _any_eval_failed(eval_scores):
            return "partial"
        return "success"
    if status in ("error", "failed", "skipped", "cancelled"):
        return "failed"
    return "partial"


def _any_eval_failed(eval_scores: Any) -> bool:
    """True if any per-spec / per-metric score is below 0.5."""
    if not isinstance(eval_scores, dict):
        return False
    for spec_scores in eval_scores.values():
        if not isinstance(spec_scores, dict):
            continue
        for metric_value in spec_scores.values():
            if isinstance(metric_value, dict):
                score = metric_value.get("score")
                if isinstance(score, (int, float)) and score < 0.5:
                    return True
            elif isinstance(metric_value, (int, float)) and metric_value < 0.5:
                return True
    return False


def _flatten_eval_scores(eval_scores: dict[str, Any]) -> dict[str, float]:
    """Average per-metric scores across all specs in the feature.

    Turns ``{spec_id: {metric: {score: 0.8}}}`` into
    ``{metric: 0.8}``. Metrics only present on some specs get
    averaged over the specs that had them.
    """
    buckets: dict[str, list[float]] = {}
    for spec_scores in eval_scores.values():
        if not isinstance(spec_scores, dict):
            continue
        for metric, value in spec_scores.items():
            if isinstance(value, dict):
                score = value.get("score")
            else:
                score = value
            if isinstance(score, (int, float)):
                buckets.setdefault(metric, []).append(float(score))
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


def _extract_tool_call_counts(stats: dict[str, Any]) -> dict[str, int]:
    """Pull a ``tool_name → count`` dict out of the swarm stats.

    The orchestrator's stats shape puts counts in various places
    depending on the swarm implementation. We try a few conventional
    keys and return an empty dict if nothing's there.
    """
    for key in ("tool_call_counts", "tool_calls_by_name", "tool_calls"):
        value = stats.get(key)
        if isinstance(value, dict):
            # Only keep integer-valued entries (some shapes put
            # timestamps here).
            return {k: int(v) for k, v in value.items() if isinstance(v, (int, float))}
    return {}
