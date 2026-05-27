"""DebateState — the TypedDict carried through the LangGraph subgraph.

Append-only reducers keep the state immutable-by-convention: each node
returns a delta dict that LangGraph merges into the existing state.
For fields that accumulate (``draft_history``, ``errors``), we use
``operator.add`` reducers so multiple writes inside one round merge
safely under parallel critic fan-out.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


def merge_round_lists(
    left: dict[int, list] | None,
    right: dict[int, list] | None,
) -> dict[int, list]:
    """Reducer for ``critiques_by_round`` — merge two round→list maps
    by appending lists per matching round number.

    Used as the barrier reducer under Send-based critic fan-out: each
    of the 4 parallel critic branches emits ``{round_num: [one]}`` and
    LangGraph invokes this reducer to merge them into
    ``{round_num: [c1, c2, c3, c4]}`` before the synthesize node runs.
    """

    if left is None:
        return dict(right or {})
    if right is None:
        return dict(left)
    merged: dict[int, list] = {k: list(v) for k, v in left.items()}
    for k, v in right.items():
        merged.setdefault(k, []).extend(v)
    return merged


def merge_round_dicts(
    left: dict[int, dict] | None,
    right: dict[int, dict] | None,
) -> dict[int, dict]:
    """Reducer for ``rebuttals_by_round`` / ``scores_by_round`` — takes
    the right-hand value on conflict (most-recent-wins)."""

    if left is None:
        return dict(right or {})
    if right is None:
        return dict(left)
    merged: dict[int, dict] = dict(left)
    merged.update(right)
    return merged


class DraftRecord(TypedDict, total=False):
    """One entry in ``draft_history`` — produced by generator or synthesizer."""

    round: int
    refined: dict                # Draft.model_dump()
    author_role: str             # "product" (v0) | "judge" (v1+)
    model_tier: str              # resolved model id at the time of authoring


class DebateState(TypedDict, total=False):
    # ── Immutable inputs set once at invocation ──────────────────────
    requirement: dict                    # RawRequirement.model_dump()
    all_requirements: list[dict]
    run_context: dict | None
    refinery_run_id: str

    # Debate parameters (copied from PipelineConfig at entry)
    max_rounds: int                      # default 3
    score_threshold: float               # default 0.8
    research_call_cap: int               # default 1
    escalation_cap: int                  # default 1
    base_model: str
    strong_model: str
    reasoning_effort: str

    # Shared retrieval / runner plumbing (opaque bag until Phase 6)
    evidence_bag: dict[str, Any]

    # ── Per-round accumulators ───────────────────────────────────────
    draft_history: Annotated[list[DraftRecord], operator.add]
    # Custom reducers so parallel Send branches merge without overwriting.
    critiques_by_round: Annotated[dict[int, list[dict]], merge_round_lists]
    rebuttals_by_round: Annotated[dict[int, dict], merge_round_dicts]
    scores_by_round: Annotated[dict[int, dict], merge_round_dicts]

    # Per-branch overlay set by dispatch_critics (LangGraph Send pattern):
    # each critic branch receives a copy of the state with _critic_role
    # overridden to the role it should invoke. Not part of the shared
    # terminal state.
    _critic_role: str

    # Research + escalation (used by Phases 7-8; Phase 4 leaves them empty)
    research_notes: Annotated[list[dict], operator.add]
    research_calls_used: int
    escalation_level: int

    # Routing counters
    round_number: int                    # incremented only by synthesizer

    # Suggested memories produced by debate-event hooks. Each entry is
    # a ``MemoryAuditEntry.model_dump()``. Append-only; finalize_node
    # copies the accumulated list into ``final_trace["memory_audits"]``
    # so downstream code can persist or display them.
    memory_audits: Annotated[list[dict], operator.add]

    # ── Terminal outputs populated by finalize_node ──────────────────
    final_refined: dict | None           # Draft.model_dump() chosen as best
    final_trace: dict | None             # DebateTrace.model_dump()

    # ── Resilience ───────────────────────────────────────────────────
    errors: Annotated[list[dict], operator.add]
    aborted_reason: str | None
