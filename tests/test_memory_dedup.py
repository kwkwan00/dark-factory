"""Tests for memory deduplication and the /api/metrics/memory endpoint.

Unit tests for MemoryDedupHelper and MemoryRepository dedup integration
come first, followed by the endpoint tests that require the ``api_client``
fixture.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dark_factory.memory.dedup_writer import MemoryDedupHelper


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_vector_repo(search_results=None, side_effect=None):
    """Create a mock VectorRepository that returns the given search results."""
    repo = MagicMock()
    if side_effect is not None:
        repo.search_memories.side_effect = side_effect
    else:
        repo.search_memories.return_value = search_results or []
    return repo


def _make_mock_neo4j():
    """Create a mock Neo4jClient with a session context manager."""
    client = MagicMock()
    session = MagicMock()
    client.session.return_value.__enter__ = MagicMock(return_value=session)
    client.session.return_value.__exit__ = MagicMock(return_value=False)
    return client, session


# ── Unit tests: MemoryDedupHelper ──────────────────────────────────────────


def test_dedup_helper_disabled_when_no_vector_repo():
    """Helper is disabled when vector_repo is None."""
    helper = MemoryDedupHelper(vector_repo=None)
    assert helper.enabled is False


def test_dedup_helper_disabled_when_threshold_zero():
    """Helper is disabled when threshold is set to 0.0."""
    repo = _make_vector_repo()
    helper = MemoryDedupHelper(vector_repo=repo, threshold=0.0)
    assert helper.enabled is False


def test_dedup_helper_enabled_with_repo_and_threshold():
    """Helper is enabled when both vector_repo and threshold > 0."""
    repo = _make_vector_repo()
    helper = MemoryDedupHelper(vector_repo=repo, threshold=0.92)
    assert helper.enabled is True


def test_dedup_find_existing_match_returns_none_when_disabled():
    """find_existing_match returns None when dedup is disabled."""
    helper = MemoryDedupHelper(vector_repo=None)
    result = helper.find_existing_match(
        memory_type="pattern",
        query_text="use parameterized queries",
        source_feature="auth",
    )
    assert result is None


def test_dedup_find_existing_match_returns_none_for_empty_query():
    """find_existing_match returns None for empty/whitespace query_text."""
    repo = _make_vector_repo()
    helper = MemoryDedupHelper(vector_repo=repo)
    result = helper.find_existing_match(
        memory_type="pattern",
        query_text="   ",
        source_feature="auth",
    )
    assert result is None
    repo.search_memories.assert_not_called()


def test_dedup_find_existing_match_above_threshold():
    """find_existing_match returns the top hit when score >= threshold."""
    repo = _make_vector_repo(search_results=[
        {"id": "pattern-abc", "score": 0.95, "description": "use parameterized queries"},
    ])
    helper = MemoryDedupHelper(vector_repo=repo, threshold=0.92)
    result = helper.find_existing_match(
        memory_type="pattern",
        query_text="use parameterized queries for SQL",
        source_feature="auth",
    )
    assert result is not None
    assert result["id"] == "pattern-abc"
    assert result["score"] == 0.95


def test_dedup_find_existing_match_below_threshold():
    """find_existing_match returns None when top score < threshold."""
    repo = _make_vector_repo(search_results=[
        {"id": "pattern-abc", "score": 0.85, "description": "use prepared statements"},
    ])
    helper = MemoryDedupHelper(vector_repo=repo, threshold=0.92)
    result = helper.find_existing_match(
        memory_type="pattern",
        query_text="use parameterized queries for SQL",
        source_feature="auth",
    )
    assert result is None


def test_dedup_cross_feature_opt_in():
    """match_cross_feature=True passes source_feature=None to search."""
    repo = _make_vector_repo(search_results=[])
    helper = MemoryDedupHelper(vector_repo=repo, threshold=0.92)
    helper.find_existing_match(
        memory_type="pattern",
        query_text="use parameterized queries",
        source_feature="auth",
        match_cross_feature=True,
    )
    call_kwargs = repo.search_memories.call_args.kwargs
    assert call_kwargs["source_feature"] is None


def test_dedup_same_feature_default():
    """By default, search is scoped to the same source_feature."""
    repo = _make_vector_repo(search_results=[])
    helper = MemoryDedupHelper(vector_repo=repo, threshold=0.92)
    helper.find_existing_match(
        memory_type="pattern",
        query_text="use parameterized queries",
        source_feature="auth",
        match_cross_feature=False,
    )
    call_kwargs = repo.search_memories.call_args.kwargs
    assert call_kwargs["source_feature"] == "auth"


def test_dedup_graceful_on_search_failure():
    """Search failures return None (graceful fallback)."""
    repo = _make_vector_repo(side_effect=RuntimeError("qdrant down"))
    helper = MemoryDedupHelper(vector_repo=repo, threshold=0.92)
    result = helper.find_existing_match(
        memory_type="pattern",
        query_text="use parameterized queries",
        source_feature="auth",
    )
    assert result is None


def test_dedup_handles_non_list_results():
    """Non-list results from search are treated as no match."""
    repo = _make_vector_repo()
    # MagicMock returns a MagicMock by default, which is not a list
    repo.search_memories.return_value = MagicMock()
    helper = MemoryDedupHelper(vector_repo=repo, threshold=0.92)
    result = helper.find_existing_match(
        memory_type="pattern",
        query_text="use parameterized queries",
        source_feature="auth",
    )
    assert result is None


def test_dedup_handles_non_dict_top_result():
    """If the top result in the list is not a dict, return None."""
    repo = _make_vector_repo(search_results=["not-a-dict"])
    helper = MemoryDedupHelper(vector_repo=repo, threshold=0.92)
    result = helper.find_existing_match(
        memory_type="pattern",
        query_text="use parameterized queries",
        source_feature="auth",
    )
    assert result is None


def test_dedup_handles_invalid_score():
    """If the score field is not a number, return None."""
    repo = _make_vector_repo(search_results=[
        {"id": "pattern-abc", "score": "not-a-number"},
    ])
    helper = MemoryDedupHelper(vector_repo=repo, threshold=0.92)
    result = helper.find_existing_match(
        memory_type="pattern",
        query_text="use parameterized queries",
        source_feature="auth",
    )
    assert result is None


def test_dedup_handles_none_score():
    """If the score field is None, return None."""
    repo = _make_vector_repo(search_results=[
        {"id": "pattern-abc", "score": None},
    ])
    helper = MemoryDedupHelper(vector_repo=repo, threshold=0.92)
    result = helper.find_existing_match(
        memory_type="pattern",
        query_text="test query",
        source_feature="auth",
    )
    assert result is None


def test_dedup_record_pattern_dedupes(monkeypatch):
    """MemoryRepository.record_pattern returns existing id on dedup match."""
    from dark_factory.memory.repository import MemoryRepository

    neo4j_client, session = _make_mock_neo4j()
    vector_repo = _make_vector_repo(search_results=[
        {"id": "pattern-existing", "score": 0.95},
    ])
    repo = MemoryRepository(neo4j_client, vector_repo, dedup_threshold=0.92)
    result = repo.record_pattern(
        description="use parameterized queries",
        context="SQL injection prevention",
        source_feature="auth",
        agent="Coder",
    )
    assert result == "pattern-existing"


def test_dedup_record_mistake_dedupes():
    """MemoryRepository.record_mistake returns existing id on dedup match."""
    from dark_factory.memory.repository import MemoryRepository

    neo4j_client, session = _make_mock_neo4j()
    vector_repo = _make_vector_repo(search_results=[
        {"id": "mistake-existing", "score": 0.95},
    ])
    repo = MemoryRepository(neo4j_client, vector_repo, dedup_threshold=0.92)
    result = repo.record_mistake(
        description="forgot to close file handle",
        error_type="ResourceLeak",
        trigger_context="open() without context manager",
        source_feature="io",
        agent="Reviewer",
    )
    assert result == "mistake-existing"


def test_dedup_record_solution_dedupes():
    """MemoryRepository.record_solution returns existing id on dedup match."""
    from dark_factory.memory.repository import MemoryRepository

    neo4j_client, session = _make_mock_neo4j()
    vector_repo = _make_vector_repo(search_results=[
        {"id": "solution-existing", "score": 0.95},
    ])
    repo = MemoryRepository(neo4j_client, vector_repo, dedup_threshold=0.92)
    result = repo.record_solution(
        description="use context manager for files",
        source_feature="io",
        agent="Coder",
    )
    assert result == "solution-existing"


def test_dedup_record_strategy_dedupes():
    """MemoryRepository.record_strategy returns existing id on dedup match."""
    from dark_factory.memory.repository import MemoryRepository

    neo4j_client, session = _make_mock_neo4j()
    vector_repo = _make_vector_repo(search_results=[
        {"id": "strategy-existing", "score": 0.95},
    ])
    repo = MemoryRepository(neo4j_client, vector_repo, dedup_threshold=0.92)
    result = repo.record_strategy(
        description="always validate inputs at boundaries",
        applicability="any function accepting external data",
        source_feature="security",
        agent="Architect",
    )
    assert result == "strategy-existing"


def test_boost_relevance_calls_neo4j():
    """boost_relevance runs Cypher via execute_write (or run) for the given label."""
    from dark_factory.memory.repository import MemoryRepository

    neo4j_client, session = _make_mock_neo4j()
    repo = MemoryRepository(neo4j_client)
    repo.boost_relevance("pattern-abc", "Pattern", delta=0.1)
    # The implementation prefers execute_write when available (MagicMock has it)
    if session.execute_write.called:
        session.execute_write.assert_called_once()
    else:
        session.run.assert_called_once()


def test_demote_relevance_calls_neo4j():
    """demote_relevance runs Cypher via execute_write (or run) for the given label."""
    from dark_factory.memory.repository import MemoryRepository

    neo4j_client, session = _make_mock_neo4j()
    repo = MemoryRepository(neo4j_client)
    repo.demote_relevance("pattern-abc", "Pattern", delta=0.05)
    if session.execute_write.called:
        session.execute_write.assert_called_once()
    else:
        session.run.assert_called_once()


def test_boost_invalid_label_is_noop():
    """boost_relevance with an invalid label does nothing."""
    from dark_factory.memory.repository import MemoryRepository

    neo4j_client, session = _make_mock_neo4j()
    repo = MemoryRepository(neo4j_client)
    repo.boost_relevance("pattern-abc", "InvalidLabel", delta=0.1)
    session.run.assert_not_called()


def test_demote_invalid_label_is_noop():
    """demote_relevance with an invalid label does nothing."""
    from dark_factory.memory.repository import MemoryRepository

    neo4j_client, session = _make_mock_neo4j()
    repo = MemoryRepository(neo4j_client)
    repo.demote_relevance("pattern-abc", "InvalidLabel", delta=0.05)
    session.run.assert_not_called()


def test_dedup_metrics_emitted_on_write():
    """observe_memory_write is called with outcome=created on new writes."""
    from dark_factory.memory.repository import MemoryRepository

    neo4j_client, session = _make_mock_neo4j()
    # No dedup match — force creation path
    vector_repo = _make_vector_repo(search_results=[])
    repo = MemoryRepository(neo4j_client, vector_repo, dedup_threshold=0.92)

    with patch("dark_factory.metrics.prometheus.observe_memory_write") as mock_write:
        repo.record_pattern(
            description="new pattern",
            context="some context",
            source_feature="auth",
            agent="Coder",
        )
        mock_write.assert_called_once_with(memory_type="pattern", outcome="created")


def test_set_dedup_threshold_clamps():
    """set_dedup_threshold clamps values to [0.0, 1.0]."""
    from dark_factory.memory.repository import MemoryRepository

    neo4j_client, _ = _make_mock_neo4j()
    repo = MemoryRepository(neo4j_client)
    repo.set_dedup_threshold(1.5)
    assert repo.dedup_helper.threshold == 1.0
    repo.set_dedup_threshold(-0.5)
    assert repo.dedup_helper.threshold == 0.0
    repo.set_dedup_threshold(0.85)
    assert repo.dedup_helper.threshold == 0.85


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
    """Repo errors on stats query -> HTTP 503 (it's the load-bearing
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
