"""T0 — Structured Knowledge (Neo4j + Qdrant memory).

Highest trust tier. Wraps the existing ``MemoryRepository`` +
``GraphRepository`` so research queries against internal state don't
need fresh embeddings or re-indexing.
"""

from __future__ import annotations

from typing import Any, ClassVar

import structlog

from dark_factory.api.refinery.contracts import Source, SourceTier
from dark_factory.api.refinery.research.base import (
    ResearchProvider,
    TierBudgetExhausted,
)

log = structlog.get_logger()


class InternalKnowledgeProvider(ResearchProvider):
    """Searches Neo4j / Qdrant memory via the existing MemoryRepository."""

    tier: ClassVar[SourceTier] = SourceTier.T0_STRUCTURED
    provider_name: ClassVar[str] = "internal-knowledge"

    def __init__(
        self,
        *,
        memory_repo: Any | None = None,
        vector_repo: Any | None = None,
    ) -> None:
        self._memory = memory_repo
        self._vector = vector_repo
        self._calls = 0
        self._budget = 0
        self._fetched_ids: set[str] = set()

    def reset_budget(self, budget: int) -> None:
        self._budget = budget
        self._calls = 0

    def search(self, query: str, budget: int) -> list[Source]:
        if self._calls >= self._budget:
            raise TierBudgetExhausted(
                f"T0 budget {self._budget} exhausted after {self._calls} calls"
            )
        self._calls += 1
        sources: list[Source] = []

        if self._vector is not None:
            try:
                rows = self._vector.search_memories(
                    query_text=query, limit=min(budget, 8),
                )
            except Exception as exc:
                log.warning("t0_vector_search_failed", error=str(exc))
                rows = []
            for r in rows:
                source_id = f"qdrant:{r.get('id', '')}"
                sources.append(Source(
                    id=source_id,
                    tier=self.tier,
                    url=None,
                    title=str(r.get("description") or r.get("title") or "")[:200],
                    chunk=str(r.get("description") or "")[:2000],
                    provider=self.provider_name,
                ))
                self._fetched_ids.add(source_id)

        return sources

    def fetch(self, url: str) -> str:
        # T0 content is already embedded in the ``chunk`` field of the
        # Source returned by search(). There's nothing to fetch.
        raise NotImplementedError(
            "T0 sources carry content in-band; fetch() is not supported"
        )
