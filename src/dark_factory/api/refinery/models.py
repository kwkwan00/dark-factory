"""Pydantic models for the Requirements Refinery."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RequirementRelationship(BaseModel):
    target_id: str
    # depends_on | conflicts_with | extends | replaces | related_to | supersedes
    type: str
    rationale: str


class SuggestedSpec(BaseModel):
    title: str
    capability: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)


class RefinedRequirement(BaseModel):
    id: str
    original_title: str
    original_description: str
    title: str
    description: str
    priority: str
    tags: list[str] = Field(default_factory=list)
    relationships: list[RequirementRelationship] = Field(default_factory=list)
    suggested_specs: list[SuggestedSpec] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)
    pass_context: str = ""
    # Convergence metadata from the debate graph. Optional — payloads
    # archived before convergence tracking landed (or carry-forward
    # paths that skipped the panel) leave these None so the storage
    # format and frontend stay backward compatible.
    convergence_status: str | None = None  # "converged" | "short_circuited" | "aborted"
    # Continuous convergence signal so the UI can show how close the debate
    # got, not just the terminal status:
    #   0.0  → no debate occurred (carry-forward path)
    #   1.0  → converged (Judge's score passed the threshold)
    #   else → debated but not converged; equals
    #          ``min(final_overall_score / overall_threshold, 0.99)``
    # so operators can gauge how much more detail a requirement needs.
    convergence_score: float = 0.0
    unresolved_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    explicit_tradeoffs: list[str] = Field(default_factory=list)
    # Optional summary of the LangGraph debate. Populated by the runner from
    # ``final_trace``; archived runs that pre-date debate tracing leave it
    # None. Carries: ``rounds_executed``, ``critiques_by_round`` (round_number
    # → list of {role, severity, dimension, finding, proposed_fix}),
    # ``rebuttals_by_round`` (round_number → {accepted, rejected}),
    # ``scores_by_round`` (round_number → {overall, passed, dimensions}).
    # Keeps the wire format flat-dict so the frontend can render without
    # importing the typed trace contracts.
    debate: dict | None = None


class SuggestedMemory(BaseModel):
    type: str  # pattern | strategy (legacy) — superseded by ``kind`` when present
    description: str
    context: str = ""
    applicability: str = ""
    source_feature: str = ""
    rationale: str = ""
    # Forward-compat with the V2 curated-kind memory contract. Optional so
    # existing storage payloads and legacy callers (Product role's LLM
    # extraction) keep validating; producers wired through
    # ``DebateTrace.memory_audits`` populate them so the frontend can bucket
    # by kind without a wire-format break. ``MemoryKind`` is the
    # authoritative list — currently: decision | incident | pattern |
    # constraint | conflict | hypothesis | anti_pattern. Stays ``str | None``
    # rather than ``Literal`` so old archived runs keep validating after
    # taxonomy growth; new code should validate against ``MemoryKind``
    # before stamping this field.
    kind: str | None = None
    validation_status: str | None = None  # validated | produced | unvalidated | user_dismissed
    summary: str = ""
    source_role: str = ""
    source_requirement_id: str | None = None


class RefineryResponse(BaseModel):
    summary: str
    pass_summaries: list[str] = Field(default_factory=list)
    refined_requirements: list[RefinedRequirement] = Field(default_factory=list)
    suggested_memories: list[SuggestedMemory] = Field(default_factory=list)
    new_relationships_count: int = 0
    requirements_modified_count: int = 0
    requirements_unchanged_count: int = 0
    source_run_id: str | None = None
    methodology: str = ""
    evidence_summary: str = ""
    risk_areas: list[str] = Field(default_factory=list)


class RequirementPatchRequest(BaseModel):
    """Partial update for a requirement node."""

    title: str | None = None
    description: str | None = None
    priority: str | None = Field(None, pattern=r"^(low|medium|high|critical)$")
    tags: list[str] | None = None


class ReconciliationReport(BaseModel):
    """Parsed output from the reconciliation agent."""

    duplicate_pairs: list[dict] = Field(default_factory=list)
    coherence_issues: list[dict] = Field(default_factory=list)
    relationship_fixes: list[dict] = Field(default_factory=list)
    priority_changes: list[dict] = Field(default_factory=list)
    spec_overlaps: list[dict] = Field(default_factory=list)
    summary: str = ""
