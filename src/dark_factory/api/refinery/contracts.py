"""Shared Pydantic contracts for the adversarial refinery (Parnas modules).

These are the ONLY shared types between the orchestrator and the role modules.
Role modules must not leak their internal state through these types — each role
keeps its prompt, model, retrieval strategy, and decoding parameters private
behind its RoleAgent interface.

Flow through the debate (one requirement):

    RawRequirement
        → ProductRole.propose(req, ctx)               → Draft v0
        → [Eng, Sec, Ops, Cost].critique(draft, ctx)   → [Critique]
        → JudgeRole.defend(draft, critiques, ctx)      → Rebuttal (wraps Draft v1)
        → JudgeRole.score(draft, ctx, trace)           → EvaluationScore
        → (loop until converged or short-circuited)
        → JudgeRole.reconcile_unresolved(...)          → Rebuttal (on short-circuit)
        → JudgeRole.review_set(set, traces, ctx)       → CrossReviewReport
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────
# Enums — stable vocabularies shared across roles
# ─────────────────────────────────────────────────────────────────────


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


class CritiqueDimension(str, Enum):
    CLARITY = "clarity"
    TESTABILITY = "testability"
    FEASIBILITY = "feasibility"
    COMPLETENESS = "completeness"
    RISK_COVERAGE = "risk_coverage"
    # How hard would this requirement be to undo if it ships and turns
    # out wrong? Distinct from risk_coverage (what could go wrong) and
    # feasibility (can it be built). Captures blast radius of
    # rolling back: data migrations, public API surface, contractual
    # commitments, customer-visible UX shifts.
    REVERSIBILITY = "reversibility"


class ConvergenceStatus(str, Enum):
    CONVERGED = "converged"
    SHORT_CIRCUITED = "short_circuited"
    ABORTED = "aborted"


class MemoryKind(str, Enum):
    DECISION = "decision"
    INCIDENT = "incident"
    PATTERN = "pattern"
    CONSTRAINT = "constraint"
    CONFLICT = "conflict"
    # Untested guess flagged by the panel for future verification —
    # distinct from DECISION (locked-in) and CONFLICT (recorded
    # tradeoff). Hypothesis memories carry an open question the next
    # debate's Research agent can target as a Librarian query.
    HYPOTHESIS = "hypothesis"
    # Recurring negative guidance — "don't structure X this way."
    # Distinct from INCIDENT (point-event failure) and PATTERN (positive
    # recurrent guidance). Anti-patterns capture *what to avoid*, ideally
    # paired with the recommended alternative so future Product drafts
    # don't re-litigate the same rejected approach.
    ANTI_PATTERN = "anti_pattern"


class ValidationStatus(str, Enum):
    """How the Judge classified a suggested memory for write-back purposes."""

    VALIDATED = "validated"          # Judge's final defend cited it AND debate converged
    PRODUCED = "produced"            # role-emitted during a converged debate
    UNVALIDATED = "unvalidated"      # suggested but unused, or debate did not converge
    USER_DISMISSED = "user_dismissed"


class SourceTier(int, Enum):
    """Research agent source tiers. Lower number = higher trust."""

    T0_STRUCTURED = 0          # Neo4j graph + Qdrant memory + trace store
    T1_INTERNAL = 1            # PRDs, ADRs, postmortems, runbooks
    T2_OBSERVABILITY = 2       # Prometheus + Postgres metrics + run stats
    T3_OFFICIAL = 3            # cloud / framework / vendor documentation
    T4_ACADEMIC = 4            # arXiv and conference proceedings
    T5_WEB = 5                 # general web and Q&A communities


# ─────────────────────────────────────────────────────────────────────
# Raw inputs and the shared RoleContext envelope
# ─────────────────────────────────────────────────────────────────────


class RawRequirement(BaseModel):
    """Debate input — the normalized ``req`` dict the orchestrator
    hands to ``run_phase2_generator`` (see ``refinery/stream.py``)."""

    id: str
    title: str
    description: str
    priority: str = "medium"
    tags: list[str] = Field(default_factory=list)
    source_file: str = ""


class SourceRef(BaseModel):
    """Provenance stamp for every retrieved row that enters a RoleContext."""

    collection: str              # qdrant collection name or "neo4j" / "docs"
    point_id: str                # stable id
    score: float                 # similarity or RRF score
    matched_filter_fields: list[str] = Field(default_factory=list)


class RoleContext(BaseModel):
    """Per-invocation context envelope. ``frozen=True`` blocks attribute
    *reassignment* (``ctx.evidence = {}`` raises) but Pydantic does not
    deep-freeze the contained dict / list values — in-place mutation
    (``ctx.evidence["x"] = y``) is still possible. The "no smuggling"
    discipline is therefore a code-review convention, not an enforced
    invariant; the load-bearing protection is the role-filter policy.

    Filtering is enforced at the data layer (Qdrant Filter + Neo4j WHERE)
    via ``role_slices.ROLE_FILTERS`` — the builder translates the role's
    policy into server-side predicates before the query runs, so an
    out-of-slice row cannot be returned even if the role's prompt asks
    for one.
    """

    model_config = ConfigDict(frozen=True)

    role: str                                    # product | engineering | ...
    requirement_id: str
    round_number: int = 0
    narrative: str = ""                          # LLM-ready markdown
    # Opaque evidence bag — keys are documented by the contract
    # between the orchestrator and each role. ``HybridContextBuilder``
    # stamps ``source_audit`` + ``filter_policy`` + ``context_fingerprint``
    # so the role still cannot see what it was filtered out of.
    evidence: dict[str, Any] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)             # stable ids for trace
    source_audit: list[SourceRef] = Field(default_factory=list)
    filter_policy: dict[str, Any] = Field(default_factory=dict)    # RoleFilterPolicy.model_dump()
    context_fingerprint: str = ""                                  # hash(policy + query)
    empty_reason: str | None = None                                # research / judge by design


# ─────────────────────────────────────────────────────────────────────
# Draft, Critique, Rebuttal — the verb outputs
# ─────────────────────────────────────────────────────────────────────


class DraftRelationship(BaseModel):
    target_id: str
    # depends_on | conflicts_with | extends | replaces | related_to | supersedes
    # ``supersedes`` differs from ``replaces`` in that it preserves the
    # lineage edge — operators can trace which old requirement a newer
    # one supplanted, where ``replaces`` (used by the duplicate-marker
    # path) implies semantic equivalence rather than evolution.
    type: str
    rationale: str


class DraftSpec(BaseModel):
    title: str
    capability: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)


class Draft(BaseModel):
    """Generator / synthesizer output. Superset of ``RefinedRequirement`` with
    provenance + convergence metadata. Maps back to ``RefinedRequirement`` at
    finalize time so storage and the API response stay backwards-compatible."""

    requirement_id: str
    title: str
    description: str
    priority: str
    tags: list[str] = Field(default_factory=list)
    suggested_specs: list[DraftSpec] = Field(default_factory=list)
    relationships: list[DraftRelationship] = Field(default_factory=list)
    produced_by: str             # role_name of generator or synthesizer
    iteration: int = 0           # 0 = first draft, increments after each rebuttal
    # Convergence metadata — populated by Judge.defend / reconcile_unresolved
    convergence_status: ConvergenceStatus | None = None
    unresolved_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    explicit_tradeoffs: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    """One critic's verdict on one draft. The rubber-stamp guardrail lives
    in a separate Pydantic validator registered in ``roles/base.py`` at
    the refinery boundary — it is intentionally not enforced on this bare
    schema so that tests and migration compatibility stay simple."""

    author_role: str
    dimension: CritiqueDimension
    severity: Severity
    finding: str
    proposed_fix: str = ""
    cited_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Error placeholder — populated when a critic crash leaves us with no
    # real critique. The round continues; the trace records the failure.
    error: str | None = None


class RebuttalEntry(BaseModel):
    """Decision ledger entry — which critique the Judge accepted/rejected/
    deferred, and the value-judgment behind it."""

    critique_ref: str                   # index or id into the round's critiques
    action: Literal["accepted", "rejected", "deferred", "partial"]
    rationale: str


class Rebuttal(BaseModel):
    """Judge.defend / Judge.reconcile_unresolved output.

    Carries a revised Draft plus a ledger of what the Judge did with each
    critique. The adversarial-contract rule (rule_no_rubber_stamp_rebuttal)
    enforces that ``entries[*].rationale`` has ≥ 25 chars on rejects."""

    revised_draft: Draft
    entries: list[RebuttalEntry] = Field(default_factory=list)
    mode: Literal["synthesize", "reconcile_unresolved"] = "synthesize"


# ─────────────────────────────────────────────────────────────────────
# Evaluation — the Judge.score + rules fused output
# ─────────────────────────────────────────────────────────────────────


class RuleViolation(BaseModel):
    """One rule's finding on one draft. Affects one ``CritiqueDimension``."""

    rule_id: str
    severity: Severity                  # only WARNING or BLOCKER observed here
    dimension: CritiqueDimension
    finding: str
    suggested_fix: str = ""
    injected_as_critique: bool = False


class EvaluationScore(BaseModel):
    """Unified verdict from the combined evaluation pipeline (Phase A rules
    → Phase B LLM with rule context → Phase C rule→dim penalties). Router
    reads only ``passed`` and signals (``missing_external_info``,
    ``disagreement_score``)."""

    dimensions: dict[str, float] = Field(default_factory=dict)              # post-penalty
    dimensions_semantic_raw: dict[str, float] = Field(default_factory=dict)  # pre-penalty LLM
    reasons: dict[str, str] = Field(default_factory=dict)
    overall: float = 0.0
    passed: bool = False
    aggregation: str = "min"
    thresholds: dict[str, float] = Field(default_factory=dict)
    overall_threshold: float = 0.8
    model_used: str = ""
    latency_seconds: float = 0.0
    # Router signals
    missing_external_info: bool = False
    disagreement_score: float = 0.0
    # Rules gate
    rule_violations: list[RuleViolation] = Field(default_factory=list)
    rule_warnings: list[RuleViolation] = Field(default_factory=list)
    rules_engine_failed: bool = False
    rules_engine_disabled: bool = False
    rules_skipped: list[str] = Field(default_factory=list)
    llm_short_circuited: bool = False
    fallback_used: bool = False


# ─────────────────────────────────────────────────────────────────────
# Cross-requirement review
# ─────────────────────────────────────────────────────────────────────


class CrossReviewReport(BaseModel):
    """Cross-requirement review output. Superset of the simpler
    ``ReconciliationReport`` at ``refinery/models.py`` (same field
    names for the structural-patch fields, plus debate-scoped
    additions) so ``apply_cross_review_report`` accepts either shape
    via its ``_ReportLike`` protocol."""

    summary: str = ""
    # Structural-patch fields (also defined on ReconciliationReport)
    duplicate_pairs: list[dict] = Field(default_factory=list)
    coherence_issues: list[dict] = Field(default_factory=list)
    relationship_fixes: list[dict] = Field(default_factory=list)
    priority_changes: list[dict] = Field(default_factory=list)
    spec_overlaps: list[dict] = Field(default_factory=list)
    # Debate-scoped additions
    set_scores: dict[str, float] = Field(default_factory=dict)
    set_reasons: dict[str, str] = Field(default_factory=dict)
    unresolved_debates: list[dict] = Field(default_factory=list)
    risk_areas: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Research agent contracts (layered sourcing)
# ─────────────────────────────────────────────────────────────────────


class Source(BaseModel):
    """A single retrieved source from one of the 6 research tiers."""

    id: str
    tier: SourceTier
    url: str | None = None              # None for T0 / T1 internal sources
    title: str = ""
    chunk: str = ""                     # fetched content, pre-chunked by provider
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provider: str = ""


class ExtractedClaim(BaseModel):
    """Analyst output — an atomic assertion tied to a single Source."""

    source_id: str                      # must match a Source.id (validator-enforced)
    claim: str
    confidence_from_source: float = Field(default=0.5, ge=0.0, le=1.0)


class ValidatedInsight(BaseModel):
    """Editor output — a cross-referenced, trust-weighted finding ready to
    cross the boundary from research into the debate."""

    kind: Literal[
        "pattern", "risk", "constraint", "tradeoff", "decision",
        "hypothesis", "anti_pattern",
    ]
    summary: str
    detail: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    source_tier_mix: list[SourceTier] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    contradicting_claims: list[str] = Field(default_factory=list)
    is_incident: bool = False


# ─────────────────────────────────────────────────────────────────────
# Institutional memory — curated kinds (see ``MemoryKind``)
# ─────────────────────────────────────────────────────────────────────


class SuggestedMemoryV2(BaseModel):
    """Refinery-v2 SuggestedMemory carrying kind + provenance + validation.

    Named ``V2`` to avoid conflict with the legacy ``SuggestedMemory`` in
    ``refinery/models.py`` during migration. The legacy model is kept as
    a narrower view of this one (type='pattern'|'strategy' only) so the
    existing frontend / storage paths keep working during the migration."""

    kind: MemoryKind
    summary: str                         # 1-line headline
    body: str                            # reusable, general (no req/spec IDs)
    context: str = ""                    # when this applies
    applicability: str = ""              # scope (capability area, stack, domain)
    source_role: str = ""                # role that proposed it
    source_requirement_id: str | None = None
    source_debate_round: int | None = None
    rationale: str = ""
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED

    # Kind-specific optional fields
    decision_alternatives: list[str] | None = None
    incident_severity: Literal["sev1", "sev2", "sev3"] | None = None
    pattern_kind: Literal["code", "requirement"] | None = None
    constraint_domain: Literal["system", "business"] | None = None
    conflict_parties: list[str] | None = None
    conflict_resolution: str | None = None
    # HYPOTHESIS-specific: the open question the panel surfaced and how
    # the next debate's Research agent should attempt to verify it.
    hypothesis_verification_query: str | None = None
    hypothesis_status: Literal["open", "verified", "refuted"] | None = None
    # ANTI_PATTERN-specific: the recommended alternative the next
    # debate should reach for instead, plus a short summary of why
    # this approach harms — so a future Product draft preempts it
    # without needing the original critique text in context.
    anti_pattern_alternative: str | None = None
    anti_pattern_harm: str | None = None

    # Research provenance (populated when source_role="research")
    provenance_source_tier_mix: list[SourceTier] | None = None
    provenance_confidence: float | None = None
    provenance_citations: list[str] | None = None
