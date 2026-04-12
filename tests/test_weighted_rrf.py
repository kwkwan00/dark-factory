"""Tests for the relevance-weighted Reciprocal Rank Fusion merge.

Tier A change: ``vector/merge.py::hybrid_merge`` now multiplies each
rank contribution by the memory's ``relevance_score`` so the
boost/demote feedback loop actually influences retrieval ordering.
Covers:

- High-relevance memory outranks low-relevance memory at the same rank
- Items without ``relevance_score`` default to 0.5 (neutral)
- 0.0 relevance floors to the minimum weight (never zero)
- Invalid / None / non-numeric relevance falls back to default
- Ordering is stable across re-runs (deterministic tiebreaker)
- Backward compatibility with pre-Tier-A items that don't carry the field
- Single-source items still included
"""

from __future__ import annotations

from dark_factory.vector.merge import (
    DEFAULT_RELEVANCE,
    MIN_RELEVANCE_WEIGHT,
    hybrid_merge,
)


def _m(id_: str, relevance: float | None = None, score: float = 0.0) -> dict:
    """Build a minimal memory row for the merge tests."""
    d: dict = {"id": id_, "score": score}
    if relevance is not None:
        d["relevance_score"] = relevance
    return d


def test_weighted_rrf_high_relevance_outranks_low_relevance():
    """With two items at the same rank (one from each source), the
    one with higher relevance_score wins."""
    neo4j = [_m("lo", relevance=0.2)]
    vector = [_m("hi", relevance=0.9)]
    result = hybrid_merge(neo4j, vector, limit=5)
    assert [r["id"] for r in result] == ["hi", "lo"]


def test_weighted_rrf_missing_relevance_defaults_to_neutral():
    """Items without ``relevance_score`` should default to 0.5 so
    pre-Tier-A data doesn't get silently deranked."""
    neo4j = [{"id": "legacy"}]  # no relevance_score
    vector = [_m("new_hot", relevance=0.9)]
    result = hybrid_merge(neo4j, vector, limit=5)
    # new_hot (0.9) > legacy (0.5 default)
    assert result[0]["id"] == "new_hot"
    assert result[1]["id"] == "legacy"


def test_weighted_rrf_zero_relevance_is_floored_not_zeroed():
    """A fully-demoted memory (relevance=0.0) should still appear
    in the recall list, just last. Zeroing it out is the wrong
    failure mode — the operator can't see stale hits in order to
    clean them up."""
    neo4j = [_m("dead", relevance=0.0)]
    vector = [_m("alive", relevance=0.5)]
    result = hybrid_merge(neo4j, vector, limit=5)
    # Both present
    assert len(result) == 2
    # alive ranks first
    assert result[0]["id"] == "alive"
    assert result[1]["id"] == "dead"


def test_weighted_rrf_invalid_relevance_falls_back_to_default():
    """Garbage values in the relevance field shouldn't crash — they
    should fall back to the DEFAULT_RELEVANCE neutral weight."""
    neo4j = [
        {"id": "garbage_str", "relevance_score": "not a number"},
        {"id": "garbage_none", "relevance_score": None},
    ]
    vector = []
    result = hybrid_merge(neo4j, vector, limit=5)
    # Both included, no crash
    ids = {r["id"] for r in result}
    assert ids == {"garbage_str", "garbage_none"}


def test_weighted_rrf_same_relevance_preserves_original_ranking():
    """When every item has the same relevance, the merge should
    produce the same ordering as the underlying RRF — no surprise
    reordering based on hidden weighting."""
    neo4j = [_m("a", relevance=0.5), _m("b", relevance=0.5)]
    vector = [_m("a", relevance=0.5), _m("b", relevance=0.5)]
    result = hybrid_merge(neo4j, vector, limit=5)
    # Items appearing in both lists should rank ahead — and "a"
    # was first in both, so it should still be first.
    assert result[0]["id"] == "a"
    assert result[1]["id"] == "b"


def test_weighted_rrf_is_deterministic_across_reruns():
    """Same inputs must produce the same output."""
    neo4j = [_m("a", 0.7), _m("b", 0.4), _m("c", 0.5)]
    vector = [_m("b", 0.4, score=0.9), _m("d", 0.8)]
    first = hybrid_merge(neo4j, vector, limit=10)
    second = hybrid_merge(neo4j, vector, limit=10)
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_weighted_rrf_single_source_items_still_included():
    """An item that appears in only one list should still be in the
    output (classic RRF behaviour). The weighting should NOT zero
    it out."""
    neo4j = [_m("neo_only", 0.6)]
    vector = [_m("vec_only", 0.6)]
    result = hybrid_merge(neo4j, vector, limit=10)
    assert {r["id"] for r in result} == {"neo_only", "vec_only"}


def test_weighted_rrf_items_in_both_lists_rank_above_single_source():
    """An item scoring in both Neo4j and Qdrant should beat items
    that only scored in one — classic RRF addition behaviour
    preserved under weighting."""
    neo4j = [
        _m("both", 0.5),
        _m("neo_only", 0.9),  # higher relevance but single source
    ]
    vector = [
        _m("both", 0.5, score=0.8),
        _m("vec_only", 0.9),
    ]
    result = hybrid_merge(neo4j, vector, limit=5)
    # "both" gets two contributions (rank 0 from each) so even at
    # mid-relevance it should beat the single-source items at 0.9
    assert result[0]["id"] == "both"


def test_min_relevance_weight_constant():
    """The floor is set to 0.1 so even relevance=0 memories get SOME
    contribution. Regression guard."""
    assert MIN_RELEVANCE_WEIGHT == 0.1
    assert DEFAULT_RELEVANCE == 0.5


def test_weighted_rrf_relevance_floors_at_minimum():
    """Negative relevance (should never happen but defensive) still
    produces a positive RRF contribution."""
    neo4j = [_m("negative", relevance=-0.5)]
    vector = [_m("normal", relevance=0.5)]
    result = hybrid_merge(neo4j, vector, limit=5)
    # Both included; negative got floored to 0.1
    assert len(result) == 2
    assert result[0]["id"] == "normal"


def test_weighted_rrf_empty_inputs():
    assert hybrid_merge([], [], limit=5) == []


def test_weighted_rrf_limit_respected():
    neo4j = [_m(f"n{i}", 0.5) for i in range(20)]
    vector = [_m(f"v{i}", 0.5) for i in range(20)]
    result = hybrid_merge(neo4j, vector, limit=3)
    assert len(result) == 3
