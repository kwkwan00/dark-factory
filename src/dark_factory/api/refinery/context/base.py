"""ContextBuilder ABC + RoleFilterPolicy.

The policy is translated into server-side predicates (Qdrant Filter,
Neo4j WHERE) before the query runs, so an out-of-slice row cannot be
returned even if a role's prompt asks for one. See the plan file's
"Role-based filtering is enforced at the data layer" section for the
five information-hiding invariants that tests must cover.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dark_factory.api.refinery.contracts import RawRequirement, RoleContext


class RoleFilterPolicy(BaseModel):
    """Declarative filter predicates for a role's hybrid retrieval."""

    model_config = ConfigDict(frozen=True)

    # Qdrant memory-collection filters (AND-joined)
    memory_kinds: list[str] | None = None                # None means any kind
    memory_types: list[str] | None = None                # legacy type (pattern/mistake/...)
    memory_source_features: list[str] | None = None
    memory_payload_tags: list[str] | None = None         # must-contain-any
    memory_exclude_tags: list[str] | None = None         # must-not-contain-any

    # Qdrant episodes-collection filters
    episode_outcomes: list[str] | None = None
    episode_agents: list[str] | None = None

    # Qdrant spec/docs-collection filters
    spec_capability_prefixes: list[str] | None = None
    doc_sources: list[str] | None = None

    # Neo4j graph traversal
    graph_relations: list[str] = Field(default_factory=list)
    graph_node_tag_must_include: list[str] | None = None
    graph_node_tag_must_exclude: list[str] | None = None

    # Retrieval budget (post-fusion, across all sources)
    row_cap: int = 12

    # Empty-by-design roles (research fetches externally; judge reads trace)
    empty: bool = False
    empty_reason: str | None = None


class RunContext(BaseModel):
    """Thin wrapper around the refinery run's traceability / evidence bag,
    passed to every ``build_context`` call. Exists so we can evolve the
    shape without touching every role module."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str | None = None
    source_mode: str = ""                # "run" | "documents" | "direct"
    round_number: int = 0
    evidence: dict[str, Any] = Field(default_factory=dict)


class ContextBuilder(ABC):
    """Hides the mix of vector + BM25 + graph + docs behind a single verb."""

    @abstractmethod
    def build_context(
        self,
        requirement: RawRequirement,
        role_name: str,
        run_context: RunContext,
    ) -> RoleContext:
        """Produce the role-sliced context. Concrete implementations look
        the role's ``RoleFilterPolicy`` up in ``role_slices.ROLE_FILTERS``
        and translate it into server-side Qdrant / Neo4j predicates."""
