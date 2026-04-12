"""Reciprocal Rank Fusion for merging Neo4j and Qdrant search results.

Tier A improvement: the RRF contribution is now multiplied by each
item's ``relevance_score`` so that the existing boost/demote feedback
loop actually influences retrieval ordering. Without this weighting,
a memory that has been repeatedly demoted (because it led to failed
evals) ranks identically to a memory that has been repeatedly boosted
(because it led to successful evals) — the feedback loop was writing
to a field nothing downstream read. Multiplying by relevance fixes
that.

The weight is floored at 0.1 so that extremely-demoted memories
(relevance ≈ 0) still appear in the recall list at all — they just
rank last. Fully zeroing them out would silently hide memories from
the agent without the operator knowing, which is the wrong failure
mode: better to surface the stale hit than to pretend it doesn't
exist.

Items without a ``relevance_score`` field default to 0.5, which is
the same neutral starting point ``MemoryRepository.record_pattern``
and friends assign to new memories. This keeps the change backward
compatible with any call site that doesn't carry the field.
"""

from __future__ import annotations


# Minimum weight applied even to fully-demoted memories. Without this
# floor, a memory at relevance=0.0 would have its RRF contribution
# zeroed out and drop off the recall list entirely — which is the
# wrong failure mode because the operator can't see stale hits in
# order to clean them up. Keep them visible, just last.
MIN_RELEVANCE_WEIGHT = 0.1

# Default relevance applied when an item lacks the field. 0.5 mirrors
# the neutral score assigned by ``MemoryRepository.record_*`` to
# freshly-created memories, so pre-Tier-A data keeps ranking the
# same as under the old unweighted RRF.
DEFAULT_RELEVANCE = 0.5


def hybrid_merge(
    neo4j_results: list[dict],
    vector_results: list[dict],
    *,
    id_key: str = "id",
    k: int = 60,
    limit: int = 10,
) -> list[dict]:
    """Merge two ranked result lists using relevance-weighted Reciprocal
    Rank Fusion.

    For each item that appears in either list, the contribution is::

        weight(item) / (k + rank)

    where ``weight(item) = max(MIN_RELEVANCE_WEIGHT, item.relevance_score)``.
    The resulting score combines:

    - **Rank signal** from both sources (same as classic RRF) — items
      that land high in both the Neo4j keyword search and the Qdrant
      vector search get the strongest contribution.
    - **Relevance signal** from the repository's feedback loop —
      memories that have been boosted by past successful evals earn a
      higher weight; memories that have been demoted earn a lower
      weight but never zero.

    Items appearing in both lists still get a natural boost (the RRF
    sum of two contributions), and single-source items are still
    included — just ranked by the combined signal.

    Args:
        neo4j_results: ranked list from Neo4j keyword / graph search.
        vector_results: ranked list from Qdrant vector search.
        id_key: field to use as the unique identifier across both
            lists. Defaults to ``"id"``.
        k: RRF constant; dampens the contribution of low-ranked
            items. 60 is the value from the original RRF paper and
            matches pre-Tier-A behaviour.
        limit: maximum number of items to return.

    Returns:
        The top ``limit`` items ordered by their combined RRF score,
        highest first.
    """
    scores: dict[str, float] = {}
    data: dict[str, dict] = {}

    def _weight(item: dict) -> float:
        raw = item.get("relevance_score", DEFAULT_RELEVANCE)
        try:
            value = float(raw) if raw is not None else DEFAULT_RELEVANCE
        except (TypeError, ValueError):
            value = DEFAULT_RELEVANCE
        # Floor keeps demoted memories visible in the recall list,
        # just pushed to the back. See MIN_RELEVANCE_WEIGHT docstring.
        return max(MIN_RELEVANCE_WEIGHT, value)

    for rank, item in enumerate(neo4j_results):
        nid = item.get(id_key, "")
        if not nid:
            continue
        scores[nid] = scores.get(nid, 0.0) + _weight(item) / (k + rank)
        data[nid] = item

    for rank, item in enumerate(vector_results):
        nid = item.get(id_key, "")
        if not nid:
            continue
        scores[nid] = scores.get(nid, 0.0) + _weight(item) / (k + rank)
        if nid not in data:
            data[nid] = item
        else:
            # Preserve the vector similarity as a debugging aid on the
            # merged record. Doesn't affect ranking — that's already
            # reflected in the rank-based RRF contribution — but lets
            # the recall tool return the raw similarity for display.
            data[nid]["vector_similarity"] = item.get("score", 0)

    ranked = sorted(scores.keys(), key=lambda nid: scores[nid], reverse=True)
    return [data[nid] for nid in ranked[:limit]]
