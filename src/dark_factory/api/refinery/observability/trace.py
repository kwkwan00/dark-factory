"""DebateTrace schema — the archival full-fidelity record of one per-requirement debate.

Also serialized to Postgres (summary rows) and Prometheus (counters +
histograms). The JSON form of the trace persists to storage alongside
``response.json`` + ``REPORT.md`` via ``refinery/storage.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from dark_factory.api.refinery.contracts import (
    ConvergenceStatus,
    Critique,
    Draft,
    EvaluationScore,
    MemoryKind,
    Rebuttal,
    Severity,
    SourceTier,
    ValidationStatus,
)


# ─────────────────────────────────────────────────────────────────────
# Per-call record (one LLM / tool invocation)
# ─────────────────────────────────────────────────────────────────────


class CallRecord(BaseModel):
    """One atomic LLM or tool call record. Each node in the debate graph
    produces at least one of these."""

    call_id: str
    role: str                               # product | engineering | ... | judge | research
    kind: str                               # propose | critique | defend | score | reconcile | research.librarian | ...
    model: str = ""
    reasoning_effort: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    latency_ms: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    cost_usd: float = 0.0
    error: str | None = None


class CriticCall(CallRecord):
    """Extension of CallRecord with critique-specific fields surfaced for
    quick access (avoids digging into the nested Critique payload)."""

    severity: Literal["blocker", "warning", "info"] = "info"
    dimension: str = ""
    finding: str = ""
    cited_evidence: list[str] = Field(default_factory=list)
    rubber_stamp_retry_count: int = 0       # 0 = passed first time


# ─────────────────────────────────────────────────────────────────────
# Per-round record
# ─────────────────────────────────────────────────────────────────────


class DecisionRecord(BaseModel):
    """Router decision at the end of a round (continue / research / escalate
    / terminate). Surfaces the signals the router used so operators can
    audit why a debate took N rounds."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    round_number: int
    kind: Literal["continue", "escalate", "research", "terminate", "reconcile", "judge_fail"]
    reason: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    next_action: str = ""


class MemoryAuditEntry(BaseModel):
    """One suggested memory's lifecycle. See plan's memory-write-back
    contract for ``validation_status`` semantics."""

    suggested_memory_id: str
    kind: MemoryKind
    source_role: str
    summary: str
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    outcome: Literal["suggested", "saved", "dismissed", "dedup_blocked"] = "suggested"
    dedup_checked: bool = False
    existing_memory_id: str | None = None
    similarity: float | None = None
    provenance_source_tier_mix: list[SourceTier] | None = None
    provenance_confidence: float | None = None
    user_decision_at: datetime | None = None


class RoundRecord(BaseModel):
    """One debate round: one generator / synthesizer call, a parallel
    critic fan-out, a score call, and (optionally) a research call."""

    round_number: int
    generator_call: CallRecord | None = None
    synthesizer_call: CallRecord | None = None
    critiques: list[CriticCall] = Field(default_factory=list)
    rebuttal: Rebuttal | None = None
    score_call: CallRecord | None = None
    scores: EvaluationScore | None = None
    research_call: CallRecord | None = None
    disagreement_score: float = 0.0
    router_decision: DecisionRecord | None = None
    duration_seconds: float = 0.0


# ─────────────────────────────────────────────────────────────────────
# The full DebateTrace
# ─────────────────────────────────────────────────────────────────────


class DebateTrace(BaseModel):
    """Full archival record of one per-requirement debate."""

    requirement_id: str
    refinery_run_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    rounds: list[RoundRecord] = Field(default_factory=list)
    final_draft: Draft | None = None
    final_score: EvaluationScore | None = None
    convergence_status: ConvergenceStatus | None = None
    termination_reason: str = ""
    decisions: list[DecisionRecord] = Field(default_factory=list)
    memory_audits: list[MemoryAuditEntry] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    research_notes: list[dict[str, Any]] = Field(default_factory=list)
    escalation_level: int = 0
    research_calls_used: int = 0
