"""Reciprocal Rank Fusion for merging Neo4j and Qdrant search results."""

from __future__ import annotations


def hybrid_merge(
    neo4j_results: list[dict],
    vector_results: list[dict],
    *,
    id_key: str = "id",
    k: int = 60,
    limit: int = 10,
) -> list[dict]:
    """Merge two ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum of 1/(k + rank) across lists where the item appears.
    Items appearing in both lists get boosted; single-source items still included.
    """
    scores: dict[str, float] = {}
    data: dict[str, dict] = {}

    for rank, item in enumerate(neo4j_results):
        nid = item.get(id_key, "")
        if not nid:
            continue
        scores[nid] = scores.get(nid, 0) + 1.0 / (k + rank)
        data[nid] = item

    for rank, item in enumerate(vector_results):
        nid = item.get(id_key, "")
        if not nid:
            continue
        scores[nid] = scores.get(nid, 0) + 1.0 / (k + rank)
        if nid not in data:
            data[nid] = item
        else:
            data[nid]["vector_similarity"] = item.get("score", 0)

    ranked = sorted(scores.keys(), key=lambda nid: scores[nid], reverse=True)
    return [data[nid] for nid in ranked[:limit]]
