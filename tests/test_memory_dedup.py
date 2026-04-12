"""Tests for the write-time memory deduplication helper + repository integration.

Tier A change: ``MemoryRepository.record_*`` methods call
``MemoryDedupHelper.find_existing_match`` before creating a new node.
Matches above threshold get boosted instead of duplicated. This file
covers:

- Threshold respect: matches at/above threshold get picked up; matches
  below are rejected.
- Type isolation: Pattern dedup never accidentally merges into a Mistake.
- Source feature filtering: dedup scopes to the same feature by default.
- Graceful degradation: Qdrant outage, embedding failure, or missing
  vector_repo all fall through to "no match" without crashing.
- Integration: record_pattern returns the existing id on match, boosts
  relevance, and skips the Neo4j CREATE.
- Integration: record_mistake bumps times_seen on the existing node on match.
- Integration: record_solution honours the mistake_id linkage even on
  dedup path.
- Disabled via threshold=0.0.
- Metric counter wiring: created / deduped outcomes fire the right counters.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dark_factory.memory.dedup_writer import MemoryDedupHelper
from dark_factory.memory.repository import MemoryRepository


# ── MemoryDedupHelper unit tests ────────────────────────────────────────────


def _make_vector_repo(search_results: list[dict]):
    vec = MagicMock()
    vec.search_memories.return_value = search_results
    return vec


def test_dedup_helper_disabled_when_no_vector_repo():
    helper = MemoryDedupHelper(vector_repo=None, threshold=0.92)
    assert helper.enabled is False
    assert (
        helper.find_existing_match(
            memory_type="pattern",
            query_text="use JWT",
            source_feature="auth",
        )
        is None
    )


def test_dedup_helper_disabled_when_threshold_zero():
    """threshold=0.0 is the explicit disable switch."""
    vec = _make_vector_repo([{"id": "p-1", "score": 1.0}])
    helper = MemoryDedupHelper(vector_repo=vec, threshold=0.0)
    assert helper.enabled is False
    # Even with a guaranteed match, no search happens
    assert (
        helper.find_existing_match(
            memory_type="pattern",
            query_text="x",
            source_feature="f",
        )
        is None
    )
    vec.search_memories.assert_not_called()


def test_dedup_helper_returns_match_when_score_above_threshold():
    vec = _make_vector_repo(
        [
            {"id": "p-existing", "score": 0.95, "description": "use JWT"},
            {"id": "p-other", "score": 0.6, "description": "unrelated"},
        ]
    )
    helper = MemoryDedupHelper(vector_repo=vec, threshold=0.92)
    match = helper.find_existing_match(
        memory_type="pattern",
        query_text="use JWT",
        source_feature="auth",
    )
    assert match is not None
    assert match["id"] == "p-existing"


def test_dedup_helper_rejects_match_below_threshold():
    vec = _make_vector_repo([{"id": "p-meh", "score": 0.80}])
    helper = MemoryDedupHelper(vector_repo=vec, threshold=0.92)
    assert (
        helper.find_existing_match(
            memory_type="pattern",
            query_text="use JWT",
            source_feature="auth",
        )
        is None
    )


def test_dedup_helper_tolerates_empty_query():
    vec = _make_vector_repo([{"id": "p-1", "score": 1.0}])
    helper = MemoryDedupHelper(vector_repo=vec, threshold=0.5)
    assert (
        helper.find_existing_match(
            memory_type="pattern",
            query_text="   ",
            source_feature="auth",
        )
        is None
    )
    vec.search_memories.assert_not_called()


def test_dedup_helper_tolerates_search_exception():
    """Qdrant outage → no match, no crash."""
    vec = MagicMock()
    vec.search_memories.side_effect = RuntimeError("qdrant down")
    helper = MemoryDedupHelper(vector_repo=vec, threshold=0.92)
    result = helper.find_existing_match(
        memory_type="pattern",
        query_text="use JWT",
        source_feature="auth",
    )
    assert result is None


def test_dedup_helper_scopes_to_source_feature_by_default():
    """The helper passes source_feature to search_memories so
    matches are scoped to the same feature by default."""
    vec = _make_vector_repo([])
    helper = MemoryDedupHelper(vector_repo=vec, threshold=0.92)
    helper.find_existing_match(
        memory_type="pattern",
        query_text="use JWT",
        source_feature="auth",
    )
    call_kwargs = vec.search_memories.call_args.kwargs
    assert call_kwargs["source_feature"] == "auth"
    assert call_kwargs["memory_type"] == "pattern"


def test_dedup_helper_cross_feature_opt_in():
    vec = _make_vector_repo([])
    helper = MemoryDedupHelper(vector_repo=vec, threshold=0.92)
    helper.find_existing_match(
        memory_type="pattern",
        query_text="use JWT",
        source_feature="auth",
        match_cross_feature=True,
    )
    call_kwargs = vec.search_memories.call_args.kwargs
    assert call_kwargs["source_feature"] is None


def test_dedup_helper_rejects_non_list_results():
    """Defensive: a MagicMock vector repo returning a MagicMock
    instead of a list shouldn't trick the helper into returning a
    bogus match."""
    vec = MagicMock()
    vec.search_memories.return_value = MagicMock()  # not a list
    helper = MemoryDedupHelper(vector_repo=vec, threshold=0.92)
    assert (
        helper.find_existing_match(
            memory_type="pattern",
            query_text="x",
            source_feature="f",
        )
        is None
    )


# ── MemoryRepository integration tests ──────────────────────────────────────


def _make_mock_neo4j():
    """Build a MagicMock Neo4jClient that captures session.run calls."""
    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_client.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_client.session.return_value.__exit__ = MagicMock(return_value=False)
    return mock_client, mock_session


def test_record_pattern_creates_new_when_no_match():
    """With no dedup match, record_pattern creates a new Neo4j node."""
    mock_client, mock_session = _make_mock_neo4j()
    mock_vector = MagicMock()
    mock_vector.search_memories.return_value = []  # no matches

    repo = MemoryRepository(mock_client, vector_repo=mock_vector, dedup_threshold=0.92)
    result_id = repo.record_pattern(
        description="use parameterized queries",
        context="SQL injection prevention",
        source_feature="auth",
        agent="coder",
    )
    assert result_id.startswith("pattern-")
    # Neo4j CREATE was executed
    assert mock_session.run.called
    # The session.run call created a pattern node
    cypher_calls = [c.args[0] for c in mock_session.run.call_args_list]
    assert any("CREATE (p:Pattern" in c for c in cypher_calls)


def test_record_pattern_returns_existing_id_on_match():
    """High-similarity match → boosts existing + returns its id."""
    mock_client, mock_session = _make_mock_neo4j()
    mock_vector = MagicMock()
    mock_vector.search_memories.return_value = [
        {"id": "pattern-existing", "score": 0.95}
    ]

    repo = MemoryRepository(mock_client, vector_repo=mock_vector, dedup_threshold=0.92)
    result_id = repo.record_pattern(
        description="use parameterized queries",
        context="SQL injection prevention",
        source_feature="auth",
        agent="coder",
    )
    assert result_id == "pattern-existing"
    # No CREATE cypher should have been run — only the boost cypher
    cypher_calls = [c.args[0] for c in mock_session.run.call_args_list]
    assert not any("CREATE (p:Pattern" in c for c in cypher_calls)
    # H3 fix: boost_relevance now goes through session.execute_write(tx_fn),
    # so the cypher lives inside the transaction function rather than as a
    # direct session.run call. Verify execute_write was invoked instead.
    assert mock_session.execute_write.called, (
        "expected boost_relevance to call session.execute_write"
    )


def test_record_pattern_below_threshold_creates_new():
    """Match exists but below threshold → still creates new."""
    mock_client, mock_session = _make_mock_neo4j()
    mock_vector = MagicMock()
    mock_vector.search_memories.return_value = [
        {"id": "pattern-weakmatch", "score": 0.75}
    ]

    repo = MemoryRepository(mock_client, vector_repo=mock_vector, dedup_threshold=0.92)
    result_id = repo.record_pattern(
        description="use parameterized queries",
        context="SQL injection prevention",
        source_feature="auth",
        agent="coder",
    )
    assert result_id.startswith("pattern-")
    assert result_id != "pattern-weakmatch"


def test_record_mistake_dedup_bumps_times_seen():
    """When a mistake dedupes to an existing node, times_seen
    should be incremented so the counter reflects 'how often we've
    tripped on this'."""
    mock_client, mock_session = _make_mock_neo4j()
    mock_vector = MagicMock()
    mock_vector.search_memories.return_value = [
        {"id": "mistake-existing", "score": 0.98}
    ]

    repo = MemoryRepository(mock_client, vector_repo=mock_vector, dedup_threshold=0.92)
    result_id = repo.record_mistake(
        description="forgot CSRF validation",
        error_type="security",
        trigger_context="session endpoint",
        source_feature="auth",
        agent="reviewer",
    )
    assert result_id == "mistake-existing"
    # times_seen increment cypher was called
    cypher_calls = [c.args[0] for c in mock_session.run.call_args_list]
    assert any("times_seen" in c and "coalesce" in c for c in cypher_calls)


def test_record_solution_preserves_mistake_linkage_on_dedup():
    """When a solution is deduped to an existing node, the caller's
    mistake_id should still produce a RESOLVED_BY edge to the
    existing solution."""
    mock_client, mock_session = _make_mock_neo4j()
    mock_vector = MagicMock()
    mock_vector.search_memories.return_value = [
        {"id": "solution-existing", "score": 0.99}
    ]

    repo = MemoryRepository(mock_client, vector_repo=mock_vector, dedup_threshold=0.92)
    result_id = repo.record_solution(
        description="add CSRF middleware",
        source_feature="auth",
        agent="reviewer",
        mistake_id="mistake-new",
    )
    assert result_id == "solution-existing"
    # The RESOLVED_BY MERGE should still have been executed
    cypher_calls = [c.args[0] for c in mock_session.run.call_args_list]
    assert any("RESOLVED_BY" in c for c in cypher_calls)


def test_boost_relevance_uses_write_transaction():
    """H3 guard: boost_relevance wraps the cypher in
    session.execute_write so Neo4j's per-node write lock
    serialises concurrent boosts on the same node."""
    mock_client, mock_session = _make_mock_neo4j()

    # Capture the transaction function passed to execute_write so
    # we can verify it would run the right cypher when invoked.
    captured: dict = {}

    def _capture_tx(tx_fn):
        fake_tx = MagicMock()
        tx_fn(fake_tx)
        captured["tx_run_calls"] = fake_tx.run.call_args_list

    mock_session.execute_write.side_effect = _capture_tx

    repo = MemoryRepository(mock_client, vector_repo=None, dedup_threshold=0.0)
    repo.boost_relevance("pattern-test-1", "Pattern", delta=0.1)

    mock_session.execute_write.assert_called_once()
    # The inner transaction should have run the Pattern boost cypher.
    assert "tx_run_calls" in captured
    assert len(captured["tx_run_calls"]) == 1
    cypher = captured["tx_run_calls"][0].args[0]
    assert "MATCH (n:Pattern" in cypher
    assert "relevance_score" in cypher


def test_demote_relevance_uses_write_transaction():
    """H3 guard: demote_relevance uses the same write-transaction
    pattern as boost_relevance."""
    mock_client, mock_session = _make_mock_neo4j()

    captured: dict = {}

    def _capture_tx(tx_fn):
        fake_tx = MagicMock()
        tx_fn(fake_tx)
        captured["tx_run_calls"] = fake_tx.run.call_args_list

    mock_session.execute_write.side_effect = _capture_tx

    repo = MemoryRepository(mock_client, vector_repo=None, dedup_threshold=0.0)
    repo.demote_relevance("mistake-test-1", "Mistake", delta=0.05)

    mock_session.execute_write.assert_called_once()
    cypher = captured["tx_run_calls"][0].args[0]
    assert "MATCH (n:Mistake" in cypher
    assert "relevance_score" in cypher


def test_boost_relevance_invalid_label_is_noop():
    """Unknown label should warn + return without hitting Neo4j."""
    mock_client, mock_session = _make_mock_neo4j()
    repo = MemoryRepository(mock_client, vector_repo=None, dedup_threshold=0.0)
    repo.boost_relevance("x", "NotARealLabel", delta=0.1)
    mock_session.execute_write.assert_not_called()


def test_record_strategy_dedup_returns_existing():
    mock_client, mock_session = _make_mock_neo4j()
    mock_vector = MagicMock()
    mock_vector.search_memories.return_value = [
        {"id": "strategy-existing", "score": 0.94}
    ]

    repo = MemoryRepository(mock_client, vector_repo=mock_vector, dedup_threshold=0.92)
    result_id = repo.record_strategy(
        description="start with JWT then fall back to sessions",
        applicability="auth features",
        source_feature="auth",
        agent="planner",
    )
    assert result_id == "strategy-existing"
    cypher_calls = [c.args[0] for c in mock_session.run.call_args_list]
    assert not any("CREATE (st:Strategy" in c for c in cypher_calls)


def test_dedup_disabled_via_zero_threshold_always_creates_new():
    """Setting the threshold to 0.0 disables dedup entirely — every
    record_* call creates a new node even when a perfect match
    exists."""
    mock_client, mock_session = _make_mock_neo4j()
    mock_vector = MagicMock()
    mock_vector.search_memories.return_value = [
        {"id": "pattern-existing", "score": 1.0}
    ]

    repo = MemoryRepository(mock_client, vector_repo=mock_vector, dedup_threshold=0.0)
    result_id = repo.record_pattern(
        description="use parameterized queries",
        context="SQL",
        source_feature="auth",
        agent="coder",
    )
    assert result_id.startswith("pattern-")
    assert result_id != "pattern-existing"
    # search_memories shouldn't even have been called — helper is disabled
    mock_vector.search_memories.assert_not_called()


def test_set_dedup_threshold_live_updates():
    mock_client, _ = _make_mock_neo4j()
    repo = MemoryRepository(mock_client, vector_repo=None, dedup_threshold=0.92)
    assert repo.dedup_helper.threshold == 0.92
    repo.set_dedup_threshold(0.80)
    assert repo.dedup_helper.threshold == 0.80
    # Clamps to [0, 1]
    repo.set_dedup_threshold(1.5)
    assert repo.dedup_helper.threshold == 1.0
    repo.set_dedup_threshold(-0.5)
    assert repo.dedup_helper.threshold == 0.0


def test_record_pattern_emits_created_metric():
    """New-node writes fire the ``outcome=created`` counter."""
    from dark_factory.metrics import prometheus as prom

    mock_client, _ = _make_mock_neo4j()
    mock_vector = MagicMock()
    mock_vector.search_memories.return_value = []

    repo = MemoryRepository(mock_client, vector_repo=mock_vector, dedup_threshold=0.92)

    before = prom.memory_writes_total.labels(type="pattern", outcome="created")._value.get()
    repo.record_pattern(
        description="test",
        context="ctx",
        source_feature="auth",
        agent="coder",
    )
    after = prom.memory_writes_total.labels(type="pattern", outcome="created")._value.get()
    assert after == before + 1


def test_record_pattern_emits_deduped_metric_on_match():
    """Dedup-matched writes fire the ``outcome=deduped`` counter."""
    from dark_factory.metrics import prometheus as prom

    mock_client, _ = _make_mock_neo4j()
    mock_vector = MagicMock()
    mock_vector.search_memories.return_value = [{"id": "pattern-existing", "score": 0.99}]

    repo = MemoryRepository(mock_client, vector_repo=mock_vector, dedup_threshold=0.92)

    before = prom.memory_writes_total.labels(type="pattern", outcome="deduped")._value.get()
    repo.record_pattern(
        description="test",
        context="ctx",
        source_feature="auth",
        agent="coder",
    )
    after = prom.memory_writes_total.labels(type="pattern", outcome="deduped")._value.get()
    assert after == before + 1


# ── /api/metrics/memory endpoint tests ──────────────────────────────────────


def test_memory_metrics_endpoint_disabled_returns_empty_payload(api_client):
    """When memory_repo is None the endpoint returns an explicit
    disabled payload with HTTP 200 rather than 503."""
    # Stash + temporarily null the app.state.memory_repo
    app = api_client.app
    original = getattr(app.state, "memory_repo", None)
    app.state.memory_repo = None
    try:
        resp = api_client.get("/api/metrics/memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["counts_by_type"] == {}
        assert data["top_recalled"] == []
        assert data["recall_effectiveness"]["total_recalls"] == 0
    finally:
        app.state.memory_repo = original


def test_memory_metrics_endpoint_reads_from_repo(api_client):
    """With a stubbed memory_repo, the endpoint returns the repo's
    stats / top-recalled / effectiveness shapes verbatim."""
    app = api_client.app
    original = getattr(app.state, "memory_repo", None)

    fake_repo = MagicMock()
    fake_repo.get_memory_stats.return_value = {
        "Pattern": {
            "count": 5,
            "mean_relevance": 0.6,
            "median_relevance": 0.5,
            "min_relevance": 0.1,
            "max_relevance": 0.9,
            "histogram": [1, 0, 1, 2, 0, 0, 1, 0, 0, 0],
        },
        "Mistake": {
            "count": 0,
            "mean_relevance": 0.0,
            "median_relevance": 0.0,
            "min_relevance": 0.0,
            "max_relevance": 0.0,
            "histogram": [0] * 10,
        },
    }
    fake_repo.get_top_recalled_memories.return_value = [
        {
            "id": "p-1",
            "description": "use parameterised queries",
            "source_feature": "auth",
            "relevance_score": 0.9,
            "times_recalled": 12,
            "times_applied": 5,
            "memory_type": "pattern",
        }
    ]
    fake_repo.get_recall_effectiveness.return_value = {
        "window_days": 7,
        "boosted": 20,
        "demoted": 5,
        "decays": 3,
        "total_recalls": 50,
        "boost_rate": 0.4,
    }
    app.state.memory_repo = fake_repo
    try:
        resp = api_client.get("/api/metrics/memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["counts_by_type"]["Pattern"]["count"] == 5
        assert data["top_recalled"][0]["id"] == "p-1"
        assert data["recall_effectiveness"]["boost_rate"] == 0.4
    finally:
        app.state.memory_repo = original


def test_memory_metrics_endpoint_propagates_stats_query_failure(api_client):
    """Repo errors on stats query → HTTP 503 (it's the load-bearing
    query; if it's broken the dashboard can't show anything useful)."""
    app = api_client.app
    original = getattr(app.state, "memory_repo", None)
    fake_repo = MagicMock()
    fake_repo.get_memory_stats.side_effect = RuntimeError("neo4j timeout")
    app.state.memory_repo = fake_repo
    try:
        resp = api_client.get("/api/metrics/memory")
        assert resp.status_code == 503
    finally:
        app.state.memory_repo = original
