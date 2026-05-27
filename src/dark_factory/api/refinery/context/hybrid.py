"""HybridContextBuilder — the role-filtered retrieval front-door.

Composes (a) dense + BM25 Qdrant search over the ``memories``
collection, (b) optional Neo4j graph traversal, and (c) (future)
documentation-source search, all filtered by the role's
``RoleFilterPolicy``. Returns a frozen ``RoleContext`` with
``filter_policy``, ``source_audit``, and ``context_fingerprint`` all
stamped on so the trace records exactly which policy produced the
context.

The Parnas discipline this file enforces:

1. A role cannot receive rows outside its slice — filtering happens at
   query time via ``policy_to_qdrant_memory_filter``.
2. A role cannot mutate its own context — ``RoleContext`` is frozen.
3. ``empty=True`` roles (Research, Judge) never touch Qdrant / Neo4j.
4. Exclusion filters translate into Qdrant ``must_not`` clauses.
5. Graph traversal respects the policy's ``graph_relations`` allowlist.

Tests in ``tests/refinery/test_role_filter_isolation.py`` lock each
invariant in.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog

from dark_factory.api.refinery.context.base import (
    ContextBuilder,
    RoleFilterPolicy,
    RunContext,
)
from dark_factory.api.refinery.context.cache import CacheKey, ContextCache
from dark_factory.api.refinery.context.rewriter import IdentityRewriter, QueryRewriter
from dark_factory.api.refinery.context.role_slices import get_role_filter_policy
from dark_factory.api.refinery.context.translators import (
    policy_allowed_graph_relations,
    policy_to_qdrant_memory_filter,
)
from dark_factory.api.refinery.contracts import (
    RawRequirement,
    RoleContext,
    SourceRef,
)
from dark_factory.log import trace_methods
from dark_factory.vector.merge import hybrid_merge

log = structlog.get_logger()


@trace_methods
class HybridContextBuilder(ContextBuilder):
    """Concrete ``ContextBuilder``. Composes vector + graph retrieval
    behind a single ``build_context`` verb.

    Retrieval backends are passed in via constructor (not auto-imported)
    so tests can swap them for stubs and the orchestrator can inject
    already-initialized repos from FastAPI state.
    """

    def __init__(
        self,
        *,
        vector_repo: Any | None = None,
        graph_repo: Any | None = None,
        cache: ContextCache | None = None,
        rewriter: QueryRewriter | None = None,
        role_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._vector = vector_repo
        self._graph = graph_repo
        self._cache = cache or ContextCache()
        self._rewriter = rewriter or IdentityRewriter()
        self._overrides = role_overrides or {}

    # ── Public: the single retrieval verb ────────────────────────────

    def build_context(
        self,
        requirement: RawRequirement,
        role_name: str,
        run_context: RunContext,
    ) -> RoleContext:
        # Cache probe first so successive rounds on the same requirement
        # don't re-query the stores for a role whose policy didn't change.
        key = CacheKey(
            requirement_id=requirement.id,
            role=role_name,
            round_number=run_context.round_number,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Empty-by-design roles short-circuit BEFORE any store is touched.
        try:
            policy = get_role_filter_policy(
                role_name, overrides=self._overrides,
            )
        except KeyError:
            # Unknown role — treat as empty (safer than guessing a policy).
            policy = RoleFilterPolicy(empty=True, empty_reason="unknown-role")

        if policy.empty:
            ctx = self._empty_context(requirement, role_name, run_context, policy)
            self._cache.set(key, ctx)
            return ctx

        # Rewrite the query for this role + round, then query stores.
        rewritten = self._rewriter.rewrite(
            requirement.description or requirement.title,
            role=role_name,
            round_number=run_context.round_number,
        )

        vector_hits = self._vector_hits(rewritten, policy) if self._vector else []
        graph_hits = self._graph_hits(requirement.id, policy) if self._graph else []

        # Simple fusion — merge by id, preserve per-source score; graph
        # hits get a 1.5x weight per plan's "graph hits get the
        # default-weighted RRF boost" rule. Phase 7 can swap this for
        # the relevance-weighted RRF helper at vector/merge.py.
        fused = _fuse_hits(
            vector_hits,
            graph_hits,
            graph_weight=1.5,
            limit=policy.row_cap,
        )

        narrative = _render_narrative(requirement, fused, role_name)
        audit = [
            SourceRef(
                collection=r.get("_source_collection", "memories"),
                point_id=str(r.get("id", "")),
                score=float(r.get("_fused_score", r.get("score", 0.0))),
                matched_filter_fields=r.get("_matched_filter_fields", []),
            )
            for r in fused
        ]
        citations = [str(r.get("id", "")) for r in fused if r.get("id")]

        ctx = RoleContext(
            role=role_name,
            requirement_id=requirement.id,
            round_number=run_context.round_number,
            narrative=narrative,
            evidence={"rows": fused, "bag": run_context.evidence},
            citations=citations,
            source_audit=audit,
            filter_policy=policy.model_dump(),
            context_fingerprint=_fingerprint(policy, rewritten),
            empty_reason=None,
        )
        self._cache.set(key, ctx)
        return ctx

    # ── Private retrieval helpers ────────────────────────────────────

    def _vector_hits(
        self,
        query_text: str,
        policy: RoleFilterPolicy,
    ) -> list[dict]:
        """Call VectorRepository.search_memories_hybrid with the policy."""

        try:
            hits = self._vector.search_memories_hybrid(
                query_text=query_text, policy=policy,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "hybrid_context_vector_failed",
                error=str(exc),
            )
            return []
        for h in hits:
            h["_source_collection"] = "memories"
        return hits

    def _graph_hits(
        self,
        requirement_id: str,
        policy: RoleFilterPolicy,
    ) -> list[dict]:
        """Traverse the Neo4j graph, respecting the policy's edge allowlist."""

        allowed_rels = policy_allowed_graph_relations(policy)
        if not allowed_rels or self._graph is None:
            return []

        # Defer to the repo. The plan-level spec is to add role-specific
        # Cypher helpers; Phase 6 wires a thin generic traversal that
        # pulls directly connected requirements matching the allowlist.
        # Phase 7 can swap in richer per-role queries when needed.
        try:
            neighbours = self._graph.get_related_requirements(
                requirement_id=requirement_id,
                relation_types=allowed_rels,
                limit=policy.row_cap,
            )
        except AttributeError:
            # Repo is a stub in tests without the helper; return empty.
            return []
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("hybrid_context_graph_failed", error=str(exc))
            return []

        rows: list[dict] = []
        for n in neighbours or []:
            rows.append({
                **n,
                "_source_collection": "neo4j",
                "_matched_filter_fields": ["graph_relations"],
            })
        return rows

    def _empty_context(
        self,
        requirement: RawRequirement,
        role_name: str,
        run_context: RunContext,
        policy: RoleFilterPolicy,
    ) -> RoleContext:
        return RoleContext(
            role=role_name,
            requirement_id=requirement.id,
            round_number=run_context.round_number,
            narrative="",
            evidence={"rows": [], "bag": run_context.evidence},
            citations=[],
            source_audit=[],
            filter_policy=policy.model_dump(),
            context_fingerprint=_fingerprint(policy, ""),
            empty_reason=policy.empty_reason or "empty-by-design",
        )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _fuse_hits(
    vector_hits: list[dict],
    graph_hits: list[dict],
    *,
    graph_weight: float = 1.5,
    limit: int = 12,
) -> list[dict]:
    """Relevance-weighted RRF across vector + graph result lists.

    Delegates to ``vector/merge.py::hybrid_merge``, which is the
    established helper used by the swarm's ``recall_episodes`` path.
    A direct graph edge is preferred over a fuzzy semantic match by
    stamping ``relevance_score = graph_weight`` on graph hits before
    fusion — ``hybrid_merge`` already multiplies RRF contributions by
    each row's relevance_score, so the weight flows through the
    existing boost/demote feedback loop.
    """

    boosted_graph = [
        {**h, "relevance_score": h.get("relevance_score") or graph_weight}
        for h in graph_hits
    ]
    return hybrid_merge(
        neo4j_results=boosted_graph,
        vector_results=vector_hits,
        id_key="id",
        limit=limit,
    )


def _render_narrative(
    requirement: RawRequirement,
    rows: list[dict],
    role_name: str,
) -> str:
    """Render a minimal markdown narrative for the role prompt.

    Phase 6 keeps this terse — Phase 7 can add role-specific section
    layouts. The current output is deterministic enough for snapshot
    tests and rich enough for a stub LLM call to reason over.
    """

    if not rows:
        return (
            f"# Context for role `{role_name}`\n\n"
            f"No supporting rows retrieved under the role's policy.\n"
        )

    lines = [f"# Context for role `{role_name}`\n"]
    lines.append(f"## Requirement\n- id: {requirement.id}")
    lines.append(f"- title: {requirement.title}")
    lines.append(f"- description: {requirement.description}\n")
    lines.append("## Retrieved rows\n")
    for r in rows:
        src = r.get("_source_collection", "memories")
        rid = r.get("id", "?")
        score = r.get("_fused_score", r.get("score", 0))
        summary = r.get("description") or r.get("summary") or r.get("title") or ""
        lines.append(f"- [{src}:{rid}] (score={score:.2f}) {summary[:200]}")
    return "\n".join(lines)


def _fingerprint(policy: RoleFilterPolicy, query_text: str) -> str:
    """Stable hash of (policy, query_text) for trace audit purposes."""

    payload = json.dumps({
        "policy": policy.model_dump(),
        "query": query_text,
    }, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
