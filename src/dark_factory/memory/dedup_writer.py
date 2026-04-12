"""Write-time deduplication for procedural memory.

The four semantic memory types (Pattern / Mistake / Solution /
Strategy) are written by every agent in the swarm. Without dedup,
a Coder that learns *"use parameterized queries for SQL"* across
five different features creates five near-identical Pattern nodes —
they all rank high for similar queries and pollute the recall list.

Tier A fix: before creating a new memory node, the repository calls
``MemoryDedupHelper.find_existing_match`` which:

1. Embeds the candidate text via the existing ``EmbeddingService``
2. Searches the ``dark_factory_memories`` Qdrant collection for
   same-type hits (optionally scoped to the same ``source_feature``
   for conservative matching)
3. Returns the single best match ≥ the dedup threshold, if any

When a match is returned, the caller boosts the existing memory's
relevance + ``times_applied`` counter instead of creating a duplicate.

The helper is **best-effort**. A Qdrant outage, an OpenAI embedding
failure, or a missing vector repo all fall through to "no match"
— the caller then creates a new node as before. Losing a dedup
opportunity is acceptable; losing a memory write because dedup
crashed is not.

Design notes:

- **Same-feature default**: we match within the same ``source_feature``
  by default. Cross-feature merges are more valuable (same pattern
  recognised across different capabilities) but risk false positives
  (two features describe similar-sounding but structurally different
  patterns). Opt-in via ``match_cross_feature=True``.
- **Per-type collections are implicit**: the existing Qdrant schema
  filters by ``memory_type`` at search time, so Pattern dedup never
  accidentally merges into a Mistake.
- **Threshold tuning**: 0.92 is the default — higher than the
  requirement dedup (0.90) because memory false-positives corrupt
  every future recall. Lower to catch more paraphrases at the risk
  of merging genuinely distinct memories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from dark_factory.vector.repository import VectorRepository

log = structlog.get_logger()


class MemoryDedupHelper:
    """Find semantically-similar existing memories before writing new ones.

    Lives at the repository layer so every memory write path
    (``record_pattern`` / ``record_mistake`` / ``record_solution`` /
    ``record_strategy``) gets dedup automatically without each
    callsite needing to know the embedding details.
    """

    def __init__(
        self,
        vector_repo: "VectorRepository | None",
        *,
        threshold: float = 0.92,
    ) -> None:
        self._vector_repo = vector_repo
        self.threshold = threshold

    @property
    def enabled(self) -> bool:
        """Dedup is active when we have a vector repo AND a non-zero
        threshold. ``threshold=0.0`` is the explicit disable switch —
        lets operators turn dedup off via Settings without removing
        the helper from the call chain."""
        return self._vector_repo is not None and self.threshold > 0.0

    def find_existing_match(
        self,
        *,
        memory_type: str,
        query_text: str,
        source_feature: str,
        match_cross_feature: bool = False,
    ) -> dict | None:
        """Return the best matching existing memory above the
        threshold, or ``None`` if no match exists (or dedup is
        disabled, or anything failed).

        The returned dict carries the Qdrant payload for the match,
        including the original ``id`` (so the caller can call
        ``boost_relevance(id)``) and ``score`` (the cosine
        similarity, for audit logging).
        """
        if not self.enabled or not query_text.strip():
            return None

        try:
            results = self._vector_repo.search_memories(
                query_text=query_text,
                memory_type=memory_type,
                source_feature=None if match_cross_feature else source_feature,
                limit=5,
            )
        except Exception as exc:
            # Graceful fallback — a Qdrant hiccup or embedding
            # failure means this write just skips dedup and creates
            # a new node, which is the correct conservative path.
            log.warning(
                "memory_dedup_search_failed",
                memory_type=memory_type,
                source_feature=source_feature,
                error=str(exc),
            )
            return None

        # Defensive: a MagicMock vector repo in unit tests returns a
        # MagicMock instead of a list, and indexing / ``.get()`` on
        # that produces more MagicMocks that LOOK truthy but carry
        # no real data. Require a concrete list of dicts before we
        # trust the result.
        if not isinstance(results, list) or not results:
            return None

        top = results[0]
        if not isinstance(top, dict):
            return None

        # Qdrant returns rows sorted by score descending. The top hit
        # is the best candidate; accept it if it clears the threshold.
        raw_score = top.get("score")
        try:
            score = float(raw_score) if raw_score is not None else 0.0
        except (TypeError, ValueError):
            return None
        if score < self.threshold:
            return None

        log.info(
            "memory_dedup_match",
            memory_type=memory_type,
            source_feature=source_feature,
            matched_id=top.get("id", ""),
            score=round(score, 4),
            threshold=self.threshold,
        )
        return top
