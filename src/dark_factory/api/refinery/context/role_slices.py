"""ROLE_FILTERS — authoritative declarative policy per role.

The dispatch table is the ONLY place where a role's default retrieval
slice is declared. The HybridContextBuilder looks a role's policy up
here and translates it into server-side Qdrant / Neo4j predicates
BEFORE running the query — an out-of-slice row cannot be returned even
if the role's prompt asks for one.

Operators override individual fields per role via
``PipelineConfig.refinery_role_filter_overrides`` — merged at registry
``get()`` time. The effective (post-merge) policy is stamped into
``RoleContext.filter_policy`` so the trace records exactly which policy
produced the context, even across per-deployment overrides.
"""

from __future__ import annotations

from typing import Any

from dark_factory.api.refinery.context.base import RoleFilterPolicy


# The institutional-memory kinds each role consumes — matches the
# memory-consumption table in the plan's "Institutional memory" section
# and the ``MemoryKind`` enum at ``contracts.py``. Research is absent
# by design (fetches externally via the research tier providers, never
# via Qdrant memory search).

ROLE_FILTERS: dict[str, RoleFilterPolicy] = {
    # Generator — preempts known tradeoffs + reuses validated structures
    # at propose time. Widest retrieval since the generator needs the
    # fullest possible picture to draft a v0.
    "product": RoleFilterPolicy(
        memory_kinds=[
            "decision", "pattern", "conflict", "hypothesis", "anti_pattern",
        ],
        memory_types=["pattern", "strategy"],
        graph_relations=["RELATED_TO", "DEPENDS_ON", "SUPERSEDES"],
        row_cap=12,
    ),

    # Engineering critic — challenges feasibility / decomposition / API
    # shape. Reads patterns + mistakes + constraints; explicitly excludes
    # security-only memories (those are Security's job).
    "engineering": RoleFilterPolicy(
        memory_kinds=[
            "pattern", "incident", "constraint", "conflict", "anti_pattern",
        ],
        memory_types=["pattern", "mistake"],
        memory_exclude_tags=["security-only", "incident-only"],
        graph_relations=["DEPENDS_ON", "IMPLEMENTS"],
        row_cap=15,
    ),

    # Security critic — wide tag include list for compliance + privacy +
    # auth + incident memories. Opus tier per plan's model matrix.
    "security": RoleFilterPolicy(
        memory_kinds=["incident", "constraint", "conflict", "anti_pattern"],
        memory_types=["pattern", "mistake", "solution"],
        memory_payload_tags=[
            "security", "auth", "privacy", "compliance", "incident",
        ],
        graph_relations=["DEPENDS_ON", "RELATED_TO"],
        row_cap=15,
    ),

    # Operations critic — runbook / observability / ops memories +
    # incident history. Flags missing telemetry + rollback gaps.
    "operations": RoleFilterPolicy(
        memory_kinds=["incident", "constraint", "pattern", "anti_pattern"],
        memory_types=["solution"],
        memory_payload_tags=[
            "ops", "observability", "runbook", "incident",
        ],
        graph_relations=["IMPLEMENTS", "RELATED_TO"],
        row_cap=12,
    ),

    # Cost critic — infra + pricing memories. Excludes security /
    # privacy tagged items (those don't inform economic estimates).
    "cost": RoleFilterPolicy(
        memory_kinds=["constraint", "pattern"],
        memory_types=["strategy"],
        memory_payload_tags=["infra", "cost", "pricing"],
        memory_exclude_tags=["security", "privacy"],
        graph_relations=["DEPENDS_ON"],
        row_cap=10,
    ),

    # Research — empty by design. The debate router invokes Research
    # separately via the tier providers (Phase 8); Research never reads
    # from the Qdrant memory collection.
    "research": RoleFilterPolicy(
        empty=True,
        empty_reason="role-fetches-externally",
        row_cap=0,
    ),

    # Judge — consumes the DebateTrace synchronously + a narrow slice of
    # Decision/Conflict memories so it can cross-reference the current
    # debate against recorded prior panel outcomes.
    "judge": RoleFilterPolicy(
        memory_kinds=["decision", "conflict", "hypothesis", "anti_pattern"],
        memory_types=["strategy"],
        graph_relations=["RELATED_TO", "SUPERSEDES"],
        row_cap=10,
    ),
}


def get_role_filter_policy(
    role_name: str,
    *,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> RoleFilterPolicy:
    """Return the effective ``RoleFilterPolicy`` for *role_name*.

    Looks the base policy up in ``ROLE_FILTERS`` and merges any
    per-role override dict from ``PipelineConfig.refinery_role_filter_overrides``.
    The merge is shallow — top-level fields on the override replace the
    base field entirely — so an operator disabling a tag filter just
    sets ``{"memory_exclude_tags": []}``.
    """

    if role_name not in ROLE_FILTERS:
        raise KeyError(f"no default RoleFilterPolicy for role '{role_name}'")
    base = ROLE_FILTERS[role_name]
    if not overrides or role_name not in overrides:
        return base

    override_dict = overrides[role_name]
    merged = base.model_dump()
    merged.update(override_dict)
    return RoleFilterPolicy.model_validate(merged)
