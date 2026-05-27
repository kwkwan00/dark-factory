"""Translators — RoleFilterPolicy → server-side predicates.

Critical Parnas invariant: these translators are the ONLY way a policy
becomes a query filter. Every refinery-path retrieval must pass through
one of them, so an out-of-slice row can't be returned even if a role's
prompt asks for one. Tests in ``tests/refinery/test_role_filter*`` lock
this property in.

Kept provider-agnostic where possible — the Qdrant-specific translator
returns a ``qdrant_client.models.Filter`` only because that's the type
the collection's ``query_points`` API expects; the Cypher translator
returns a plain string fragment.
"""

from __future__ import annotations

from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from dark_factory.api.refinery.context.base import RoleFilterPolicy


def policy_to_qdrant_memory_filter(policy: RoleFilterPolicy) -> Filter | None:
    """Build a Qdrant ``Filter`` for the ``memories`` collection.

    Translation rules:
    - ``memory_kinds`` → ``must`` ``MatchAny`` on the ``kind`` payload key.
    - ``memory_types`` → ``must`` ``MatchAny`` on ``memory_type`` (legacy
      field written by existing ``MemoryRepository.record_pattern`` etc.).
    - ``memory_source_features`` → ``must`` ``MatchAny`` on ``source_feature``.
    - ``memory_payload_tags`` → ``should`` ``MatchAny`` on each tag field
      (payload uses ``tags: list[str]``; Qdrant matches membership).
    - ``memory_exclude_tags`` → ``must_not`` ``MatchAny`` on the same.

    Returns ``None`` when the policy specifies no filters (full-open
    retrieval). Research / Judge with ``empty=True`` return ``None``
    here, but the builder short-circuits before calling this function.
    """

    must: list[FieldCondition] = []
    should: list[FieldCondition] = []
    must_not: list[FieldCondition] = []

    if policy.memory_kinds:
        must.append(FieldCondition(
            key="kind",
            match=MatchAny(any=list(policy.memory_kinds)),
        ))
    if policy.memory_types:
        must.append(FieldCondition(
            key="memory_type",
            match=MatchAny(any=list(policy.memory_types)),
        ))
    if policy.memory_source_features:
        must.append(FieldCondition(
            key="source_feature",
            match=MatchAny(any=list(policy.memory_source_features)),
        ))
    if policy.memory_payload_tags:
        # ``tags`` is a list field in the memory payload; Qdrant's
        # MatchAny on a list field matches if ANY element of the stored
        # list matches ANY of the filter values.
        should.append(FieldCondition(
            key="tags",
            match=MatchAny(any=list(policy.memory_payload_tags)),
        ))
    if policy.memory_exclude_tags:
        must_not.append(FieldCondition(
            key="tags",
            match=MatchAny(any=list(policy.memory_exclude_tags)),
        ))

    if not (must or should or must_not):
        return None
    return Filter(
        must=must or None,
        should=should or None,
        must_not=must_not or None,
    )


def policy_to_qdrant_episode_filter(policy: RoleFilterPolicy) -> Filter | None:
    """Build a Qdrant ``Filter`` for the ``episodes`` collection."""

    must: list[FieldCondition] = []
    if policy.episode_outcomes:
        must.append(FieldCondition(
            key="outcome",
            match=MatchAny(any=list(policy.episode_outcomes)),
        ))
    if policy.episode_agents:
        must.append(FieldCondition(
            key="agent",
            match=MatchAny(any=list(policy.episode_agents)),
        ))
    return Filter(must=must or None) if must else None


def policy_to_qdrant_spec_filter(policy: RoleFilterPolicy) -> Filter | None:
    """Build a Qdrant ``Filter`` for the ``specs`` collection."""

    must: list[FieldCondition] = []
    should: list[FieldCondition] = []

    if policy.spec_capability_prefixes:
        # Qdrant has no built-in prefix match — emit one MatchValue per
        # prefix as a ``should`` so the hit matches any prefix.
        for prefix in policy.spec_capability_prefixes:
            should.append(FieldCondition(
                key="capability",
                match=MatchValue(value=prefix),
            ))
    if policy.doc_sources:
        must.append(FieldCondition(
            key="doc_source",
            match=MatchAny(any=list(policy.doc_sources)),
        ))
    if not (must or should):
        return None
    return Filter(must=must or None, should=should or None)


def policy_to_cypher_where(policy: RoleFilterPolicy, *, alias: str = "n") -> str:
    """Return a Cypher WHERE fragment that filters a node by the
    policy's graph-node tag rules. Empty string when no tag predicate
    is set. The caller is responsible for AND-joining this into its
    own WHERE clause.

    Edge type filtering (``graph_relations``) is NOT encoded here —
    that's a structural part of the pattern (``MATCH (n)-[r:FOO]-(m)``)
    and must be composed by the caller.
    """

    clauses: list[str] = []
    if policy.graph_node_tag_must_include:
        joined = ", ".join(
            f"'{t}'" for t in policy.graph_node_tag_must_include
        )
        clauses.append(f"any(t IN {alias}.tags WHERE t IN [{joined}])")
    if policy.graph_node_tag_must_exclude:
        joined = ", ".join(
            f"'{t}'" for t in policy.graph_node_tag_must_exclude
        )
        clauses.append(f"none(t IN {alias}.tags WHERE t IN [{joined}])")
    return " AND ".join(clauses)


def policy_allowed_graph_relations(policy: RoleFilterPolicy) -> list[str]:
    """Return the list of edge types the policy permits for graph
    traversal. The Phase-6 HybridContextBuilder uses this to build
    relation-type clauses like ``[r:DEPENDS_ON|RELATED_TO]``.

    Empty list means "no graph traversal at all" — the builder returns
    zero graph hits in that case.
    """

    return list(policy.graph_relations or [])
