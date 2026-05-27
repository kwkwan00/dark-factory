"""Phase 6 tests — the 5 information-hiding invariants.

The plan's "Role-based filtering is enforced at the data layer" section
declares five testable invariants that prove a role cannot receive
out-of-slice data even if its prompt asks. These tests lock each one
in.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from dark_factory.api.refinery.context import (
    HybridContextBuilder,
    IdentityRewriter,
    ROLE_FILTERS,
    RoleFilterPolicy,
    RunContext,
    get_role_filter_policy,
)
from dark_factory.api.refinery.context.hybrid import _fuse_hits
from dark_factory.api.refinery.context.translators import (
    policy_allowed_graph_relations,
    policy_to_qdrant_memory_filter,
)
from dark_factory.api.refinery.contracts import RawRequirement, RoleContext
from dark_factory.config import PipelineConfig


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


def _raw(req_id: str = "req-1") -> RawRequirement:
    return RawRequirement(
        id=req_id,
        title="Session timeout policy",
        description="System must enforce a 30-minute inactivity timeout",
        priority="high",
        tags=["auth", "security"],
    )


class _FakeVector:
    """In-memory vector repo that honours the Qdrant Filter argument in
    the search_memories_hybrid call. Returns rows tagged with their
    memory_type / kind / tags so tests can assert a role's policy
    actually filtered them out."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.last_filter = "NEVER_CALLED"

    def search_memories_hybrid(self, *, query_text, policy, limit=None):
        qdrant_filter = policy_to_qdrant_memory_filter(policy)
        self.last_filter = qdrant_filter
        # Apply the filter manually — the real Qdrant server-side impl
        # does this, but for unit tests we simulate it in-process.
        hits = []
        for row in self.rows:
            if _row_matches(row, policy):
                hits.append({
                    "id": row["id"],
                    "score": row.get("score", 0.5),
                    **row,
                })
        return hits[: (limit or policy.row_cap)]


class _FakeGraph:
    """In-memory graph repo. ``get_related_requirements`` honours the
    ``relation_types`` allowlist — any node whose stored relation type
    isn't in the allowlist is invisible."""

    def __init__(self, neighbours_by_req: dict[str, list[dict]]) -> None:
        self._by_req = neighbours_by_req
        self.last_relation_types: list[str] | None = None

    def get_related_requirements(
        self, *, requirement_id, relation_types, limit=None,
    ):
        self.last_relation_types = list(relation_types)
        out = []
        for n in self._by_req.get(requirement_id, []):
            if n.get("relation_type") in relation_types:
                out.append({**n, "score": n.get("score", 1.0)})
        return out[: (limit or 100)]


def _row_matches(row: dict, policy: RoleFilterPolicy) -> bool:
    """Mirror the Qdrant-side filter semantics in-process so fake
    retrieval honours the same policy the translator would emit."""

    if policy.memory_kinds and row.get("kind") not in policy.memory_kinds:
        return False
    if policy.memory_types and row.get("memory_type") not in policy.memory_types:
        return False
    if policy.memory_source_features:
        if row.get("source_feature") not in policy.memory_source_features:
            return False
    row_tags = set(row.get("tags", []))
    if policy.memory_payload_tags:
        if not (row_tags & set(policy.memory_payload_tags)):
            return False
    if policy.memory_exclude_tags:
        if row_tags & set(policy.memory_exclude_tags):
            return False
    return True


def _corpus() -> list[dict]:
    """A deliberately diverse corpus where each row would match at
    least one role's policy and be filtered out by at least one other."""

    return [
        # Pattern memory, auth tag — Product + Engineering allowed, Cost excluded
        {"id": "mem-1", "kind": "pattern", "memory_type": "pattern",
         "tags": ["auth", "session"], "description": "OAuth2 session TTL pattern",
         "score": 0.9},
        # Incident memory, security tag — Security allowed, Cost excluded
        {"id": "mem-2", "kind": "incident", "memory_type": "mistake",
         "tags": ["security", "incident"],
         "description": "Token leak incident 2025-09", "score": 0.85},
        # Constraint memory, infra tag — Cost allowed, Security excluded
        {"id": "mem-3", "kind": "constraint", "memory_type": "strategy",
         "tags": ["infra", "cost"],
         "description": "Redis cluster capped at 50 GB per region", "score": 0.8},
        # Decision memory, conflict tag — Product + Judge allowed
        {"id": "mem-4", "kind": "decision", "memory_type": "strategy",
         "tags": ["auth"],
         "description": "Chose PKCE over implicit flow due to mobile targets",
         "score": 0.75},
        # Conflict memory — Product + Engineering + Judge allowed
        {"id": "mem-5", "kind": "conflict", "memory_type": "pattern",
         "tags": ["auth", "security"],
         "description": "Recurring debate: refresh-token lifetime",
         "score": 0.7},
        # Solution memory, ops tag — Operations allowed
        {"id": "mem-6", "kind": "pattern", "memory_type": "solution",
         "tags": ["ops", "observability"],
         "description": "SLO burn-rate alert for auth latency", "score": 0.8},
    ]


def _neighbours() -> dict[str, list[dict]]:
    """Simulated Neo4j neighbours with explicit relation_type labels."""

    return {
        "req-1": [
            {"id": "req-2", "title": "Session manager",
             "relation_type": "DEPENDS_ON", "score": 1.0},
            {"id": "req-3", "title": "Audit log",
             "relation_type": "RELATED_TO", "score": 1.0},
            {"id": "req-4", "title": "Login API",
             "relation_type": "IMPLEMENTS", "score": 1.0},
        ],
    }


def _builder(**kw):
    return HybridContextBuilder(
        vector_repo=_FakeVector(kw.get("rows", _corpus())),
        graph_repo=_FakeGraph(kw.get("neighbours", _neighbours())),
        rewriter=IdentityRewriter(),
    )


def _run_ctx(round_number: int = 0) -> RunContext:
    return RunContext(
        run_id=None, source_mode="direct", round_number=round_number,
        evidence={},
    )


# ─────────────────────────────────────────────────────────────────────
# Invariant 1 — no cross-role leakage
# ─────────────────────────────────────────────────────────────────────


def test_invariant_no_cross_role_leakage_security_only_memory():
    """A Security-only tagged memory must never appear in
    Engineering / Operations / Cost contexts."""

    corpus = [
        {"id": "sec-only", "kind": "incident", "memory_type": "mistake",
         "tags": ["security", "incident"],
         "description": "STRIDE threat model gap", "score": 0.95},
    ]
    builder = _builder(rows=corpus)

    # Cost must NOT see the security-only row (security is in its
    # exclude_tags).
    cost_ctx = builder.build_context(_raw(), "cost", _run_ctx())
    cost_ids = {s.point_id for s in cost_ctx.source_audit}
    assert "sec-only" not in cost_ids

    # Security MUST see it.
    sec_ctx = builder.build_context(_raw(), "security", _run_ctx())
    sec_ids = {s.point_id for s in sec_ctx.source_audit}
    assert "sec-only" in sec_ids


def test_invariant_no_cross_role_leakage_cost_only_memory():
    """Cost-tagged memory must appear in Cost's context only. Security
    excludes 'cost' indirectly via its payload_tags whitelist (Security
    only includes security/auth/privacy/compliance/incident)."""

    corpus = [
        {"id": "cost-only", "kind": "constraint", "memory_type": "strategy",
         "tags": ["infra", "cost", "pricing"],
         "description": "Egress cap: $500/mo", "score": 0.9},
    ]
    builder = _builder(rows=corpus)

    cost_ctx = builder.build_context(_raw(), "cost", _run_ctx())
    assert "cost-only" in {s.point_id for s in cost_ctx.source_audit}

    sec_ctx = builder.build_context(_raw(), "security", _run_ctx())
    # Security's payload_tags don't include "cost" / "infra" / "pricing",
    # so the tag must_any filter doesn't match and the row is filtered.
    assert "cost-only" not in {s.point_id for s in sec_ctx.source_audit}


# ─────────────────────────────────────────────────────────────────────
# Invariant 2 — no post-hoc filtering (policy lands on the Qdrant Filter)
# ─────────────────────────────────────────────────────────────────────


def test_invariant_no_post_hoc_filtering_qdrant_filter_is_non_none():
    """For every refinery-path role that has filters, the Qdrant Filter
    produced by the translator must be non-None. Post-hoc filtering in
    Python (building an empty Filter, then filtering results in-process)
    is the regression this test catches."""

    vector = _FakeVector([])
    builder = HybridContextBuilder(
        vector_repo=vector, graph_repo=None, rewriter=IdentityRewriter(),
    )
    for role in ("product", "engineering", "security", "operations", "cost"):
        vector.last_filter = "NEVER_CALLED"
        builder.build_context(_raw(), role, _run_ctx())
        assert vector.last_filter is not None, (
            f"role '{role}' did not apply a Qdrant Filter at query time"
        )


# ─────────────────────────────────────────────────────────────────────
# Invariant 3 — empty-by-design roles honour the contract
# ─────────────────────────────────────────────────────────────────────


def test_invariant_empty_by_design_research_never_touches_stores():
    vector = MagicMock()
    vector.search_memories_hybrid.side_effect = AssertionError(
        "Research must never call Qdrant"
    )
    graph = MagicMock()
    graph.get_related_requirements.side_effect = AssertionError(
        "Research must never call Neo4j"
    )

    builder = HybridContextBuilder(
        vector_repo=vector, graph_repo=graph, rewriter=IdentityRewriter(),
    )
    ctx = builder.build_context(_raw(), "research", _run_ctx())
    assert ctx.empty_reason == "role-fetches-externally"
    assert ctx.source_audit == []
    assert ctx.citations == []
    vector.search_memories_hybrid.assert_not_called()
    graph.get_related_requirements.assert_not_called()


def test_invariant_unknown_role_defaults_to_empty():
    """Hardening: unknown role names get an empty policy, never a
    default-open retrieval. This stops a typo / refactor from accidentally
    opening the whole store to an unconfigured role."""

    vector = MagicMock()
    vector.search_memories_hybrid.side_effect = AssertionError(
        "unknown role must never reach Qdrant"
    )
    builder = HybridContextBuilder(
        vector_repo=vector, graph_repo=None, rewriter=IdentityRewriter(),
    )
    ctx = builder.build_context(_raw(), "nonexistent-role", _run_ctx())
    assert ctx.empty_reason == "unknown-role"


# ─────────────────────────────────────────────────────────────────────
# Invariant 4 — exclusion filters actually exclude
# ─────────────────────────────────────────────────────────────────────


def test_invariant_exclusion_filters_drop_matching_rows():
    """A row tagged with one of Cost's ``memory_exclude_tags`` (security
    / privacy) must be filtered out even if it semantically matches."""

    corpus = [
        {"id": "has-security-tag", "kind": "constraint",
         "memory_type": "strategy",
         "tags": ["infra", "cost", "security"],
         "description": "Cost AND security tagged", "score": 0.95},
    ]
    builder = _builder(rows=corpus)
    cost_ctx = builder.build_context(_raw(), "cost", _run_ctx())
    assert "has-security-tag" not in {s.point_id for s in cost_ctx.source_audit}


# ─────────────────────────────────────────────────────────────────────
# Invariant 5 — graph edge allowlist enforced
# ─────────────────────────────────────────────────────────────────────


def test_invariant_graph_edges_allowlist_enforced():
    """Cost has ``graph_relations=["DEPENDS_ON"]`` — the RELATED_TO and
    IMPLEMENTS neighbours must never appear in its context."""

    graph = _FakeGraph(_neighbours())
    builder = HybridContextBuilder(
        vector_repo=_FakeVector([]), graph_repo=graph,
        rewriter=IdentityRewriter(),
    )
    cost_ctx = builder.build_context(_raw(), "cost", _run_ctx())
    ids = {s.point_id for s in cost_ctx.source_audit}
    # Only DEPENDS_ON edges allowed.
    assert graph.last_relation_types == ["DEPENDS_ON"]
    assert "req-2" in ids or ids == set()  # req-2 is DEPENDS_ON
    assert "req-3" not in ids  # RELATED_TO — not allowed
    assert "req-4" not in ids  # IMPLEMENTS — not allowed


def test_product_graph_sees_related_to_and_depends_on():
    """Product's allowlist is ["RELATED_TO", "DEPENDS_ON"]."""

    graph = _FakeGraph(_neighbours())
    builder = HybridContextBuilder(
        vector_repo=_FakeVector([]), graph_repo=graph,
        rewriter=IdentityRewriter(),
    )
    product_ctx = builder.build_context(_raw(), "product", _run_ctx())
    ids = {s.point_id for s in product_ctx.source_audit}
    assert "req-2" in ids  # DEPENDS_ON
    assert "req-3" in ids  # RELATED_TO
    assert "req-4" not in ids  # IMPLEMENTS — blocked


def test_engineering_graph_sees_depends_on_and_implements():
    """Engineering's allowlist is ["DEPENDS_ON", "IMPLEMENTS"]."""

    graph = _FakeGraph(_neighbours())
    builder = HybridContextBuilder(
        vector_repo=_FakeVector([]), graph_repo=graph,
        rewriter=IdentityRewriter(),
    )
    eng_ctx = builder.build_context(_raw(), "engineering", _run_ctx())
    ids = {s.point_id for s in eng_ctx.source_audit}
    assert "req-2" in ids  # DEPENDS_ON
    assert "req-4" in ids  # IMPLEMENTS
    assert "req-3" not in ids  # RELATED_TO — blocked


# ─────────────────────────────────────────────────────────────────────
# Frozen RoleContext invariant
# ─────────────────────────────────────────────────────────────────────


def test_role_context_remains_frozen_after_build():
    builder = _builder()
    ctx = builder.build_context(_raw(), "product", _run_ctx())
    with pytest.raises(ValidationError):
        ctx.role = "security"  # type: ignore[misc]


def test_role_context_filter_policy_fingerprint_is_stable():
    builder = _builder()
    ctx1 = builder.build_context(_raw(), "security", _run_ctx())
    # Clear the cache so we take a fresh path, not the memoised one.
    builder._cache.clear()
    ctx2 = builder.build_context(_raw(), "security", _run_ctx())
    assert ctx1.context_fingerprint == ctx2.context_fingerprint


def test_role_context_filter_policy_fingerprint_differs_per_role():
    builder = _builder()
    p = builder.build_context(_raw(), "product", _run_ctx())
    s = builder.build_context(_raw(), "security", _run_ctx())
    assert p.context_fingerprint != s.context_fingerprint


# ─────────────────────────────────────────────────────────────────────
# Operator override merging
# ─────────────────────────────────────────────────────────────────────


def test_override_widens_security_payload_tags():
    base = ROLE_FILTERS["security"]
    merged = get_role_filter_policy(
        "security",
        overrides={"security": {"memory_payload_tags": ["security", "custom-tag"]}},
    )
    assert merged.memory_payload_tags == ["security", "custom-tag"]
    # Other fields inherit the base unchanged.
    assert merged.memory_kinds == base.memory_kinds


def test_override_for_role_without_override_returns_base():
    base = ROLE_FILTERS["product"]
    merged = get_role_filter_policy("product", overrides={})
    assert merged == base


# ─────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────


def test_context_cache_is_per_requirement_round_role():
    builder = _builder()
    c1 = builder.build_context(_raw("req-1"), "security", _run_ctx(0))
    c1_again = builder.build_context(_raw("req-1"), "security", _run_ctx(0))
    assert c1 is c1_again  # identity — cache hit

    c2 = builder.build_context(_raw("req-1"), "security", _run_ctx(1))
    assert c1 is not c2

    c3 = builder.build_context(_raw("req-1"), "engineering", _run_ctx(0))
    assert c1 is not c3


def test_context_cache_clear_for_requirement():
    builder = _builder()
    builder.build_context(_raw("req-1"), "product", _run_ctx(0))
    builder.build_context(_raw("req-2"), "product", _run_ctx(0))
    builder._cache.clear_for_requirement("req-1")
    assert len(builder._cache) == 1


# ─────────────────────────────────────────────────────────────────────
# Fusion math
# ─────────────────────────────────────────────────────────────────────


def test_fuse_hits_returns_union_by_id():
    """Graph + vector hits with the same id collapse to one row."""

    vector_hits = [{"id": "a", "score": 0.5}]
    graph_hits = [{"id": "a", "score": 0.5}]
    fused = _fuse_hits(vector_hits, graph_hits, graph_weight=1.5, limit=5)
    ids = [r["id"] for r in fused]
    assert ids.count("a") == 1
    # Original per-source score is preserved for the audit fallback.
    assert fused[0].get("score") is not None


def test_fuse_hits_ranks_graph_only_above_unrelated_vector():
    """A row present only in graph with graph_weight boost should rank
    above an unrelated vector-only row with a lower intrinsic score."""

    vector_hits = [{"id": "v-low", "score": 0.1}]
    graph_hits = [{"id": "g-hi", "score": 1.0}]
    fused = _fuse_hits(vector_hits, graph_hits, graph_weight=1.5, limit=5)
    ids = [r["id"] for r in fused]
    assert ids.index("g-hi") < ids.index("v-low")


def test_fuse_hits_respects_limit():
    vector_hits = [{"id": f"v-{i}", "score": 1.0 - i * 0.1} for i in range(10)]
    fused = _fuse_hits(vector_hits, [], graph_weight=1.5, limit=3)
    assert len(fused) == 3
