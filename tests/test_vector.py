"""Tests for the Qdrant vector search integration."""

from __future__ import annotations

from dark_factory.config import QdrantConfig, Settings
from dark_factory.vector.merge import hybrid_merge


# ── Config tests ─────────────────────────────────────────────────────


def test_qdrant_config_defaults() -> None:
    config = QdrantConfig()
    assert config.url == "http://localhost:6333"
    assert config.collection_prefix == "dark_factory"
    assert config.embedding_model == "text-embedding-3-large"
    assert config.enabled is True


def test_qdrant_config_in_settings() -> None:
    settings = Settings()
    assert settings.qdrant.enabled is True
    assert settings.qdrant.collection_prefix == "dark_factory"


# ── Hybrid merge (RRF) tests ────────────────────────────────────────


def test_hybrid_merge_both_sources() -> None:
    neo4j = [{"id": "a", "desc": "neo"}, {"id": "b", "desc": "neo"}]
    vector = [{"id": "b", "score": 0.9, "desc": "vec"}, {"id": "c", "score": 0.8, "desc": "vec"}]
    merged = hybrid_merge(neo4j, vector, limit=3)
    ids = [m["id"] for m in merged]
    # "b" appears in both → highest RRF score
    assert ids[0] == "b"
    assert len(merged) <= 3


def test_hybrid_merge_neo4j_only() -> None:
    neo4j = [{"id": "a"}, {"id": "b"}]
    merged = hybrid_merge(neo4j, [], limit=5)
    assert len(merged) == 2
    assert merged[0]["id"] == "a"


def test_hybrid_merge_vector_only() -> None:
    vector = [{"id": "x", "score": 0.9}, {"id": "y", "score": 0.7}]
    merged = hybrid_merge([], vector, limit=5)
    assert len(merged) == 2
    assert merged[0]["id"] == "x"


def test_hybrid_merge_deduplication() -> None:
    neo4j = [{"id": "same", "source": "neo4j"}]
    vector = [{"id": "same", "score": 0.95, "source": "qdrant"}]
    merged = hybrid_merge(neo4j, vector, limit=5)
    assert len(merged) == 1
    assert merged[0]["id"] == "same"
    assert "vector_similarity" in merged[0]


def test_hybrid_merge_empty() -> None:
    assert hybrid_merge([], [], limit=5) == []


def test_hybrid_merge_respects_limit() -> None:
    neo4j = [{"id": f"n{i}"} for i in range(20)]
    vector = [{"id": f"v{i}", "score": 0.5} for i in range(20)]
    merged = hybrid_merge(neo4j, vector, limit=5)
    assert len(merged) == 5


# ── Graceful degradation tests ──────────────────────────────────────


def test_recall_memories_works_without_vector() -> None:
    """When _vector_repo is None, recall_memories uses Neo4j only."""
    import dark_factory.agents.tools as tools_mod
    from unittest.mock import MagicMock

    original_vector = tools_mod._vector_repo
    original_memory = tools_mod._memory_repo

    tools_mod._vector_repo = None
    mock_memory = MagicMock()
    mock_memory.get_related_memories.return_value = [{"id": "pattern-abc", "description": "test"}]
    tools_mod._memory_repo = mock_memory
    tools_mod._recalled_memory_ids = []

    try:
        result = tools_mod.recall_memories.invoke({"feature_name": "auth"})
        assert "pattern-abc" in result
    finally:
        tools_mod._vector_repo = original_vector
        tools_mod._memory_repo = original_memory


def test_search_similar_specs_disabled() -> None:
    import dark_factory.agents.tools as tools_mod

    original = tools_mod._vector_repo
    tools_mod._vector_repo = None
    try:
        result = tools_mod.search_similar_specs.invoke({"description": "auth"})
        assert "not available" in result
    finally:
        tools_mod._vector_repo = original


def test_search_similar_code_disabled() -> None:
    import dark_factory.agents.tools as tools_mod

    original = tools_mod._vector_repo
    tools_mod._vector_repo = None
    try:
        result = tools_mod.search_similar_code.invoke({"description": "login"})
        assert "not available" in result
    finally:
        tools_mod._vector_repo = original


# ── Tool presence tests ─────────────────────────────────────────────


def test_vector_search_tools_exist() -> None:
    from dark_factory.agents.tools import (
        VECTOR_SEARCH_TOOLS,
        search_similar_code,
        search_similar_specs,
    )
    assert search_similar_specs in VECTOR_SEARCH_TOOLS
    assert search_similar_code in VECTOR_SEARCH_TOOLS


def test_coder_has_vector_search_tools() -> None:
    from unittest.mock import patch

    with patch("dark_factory.agents.swarm.create_agent") as mock_create:
        mock_create.return_value = "mock"
        from dark_factory.agents.swarm import _build_coder

        _build_coder("anthropic:claude-sonnet-4-6")

        tools = mock_create.call_args.kwargs.get("tools") or mock_create.call_args[0][1]
        names = [getattr(t, "name", str(t)) for t in tools]
        assert "search_similar_specs" in names
        assert "search_similar_code" in names


# ── Dual-write test ─────────────────────────────────────────────────


def test_memory_repo_dual_writes() -> None:
    """When vector_repo is provided, record_pattern calls both Neo4j + Qdrant."""
    from unittest.mock import MagicMock, patch

    from dark_factory.memory.repository import MemoryRepository

    mock_neo4j = MagicMock()
    mock_session = MagicMock()
    mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mock_vector = MagicMock()

    repo = MemoryRepository(mock_neo4j, vector_repo=mock_vector)
    repo.record_pattern(
        description="test pattern",
        context="test context",
        source_feature="auth",
        agent="coder",
    )

    # Neo4j was called
    mock_session.run.assert_called()
    # Qdrant was called
    mock_vector.upsert_memory.assert_called_once()
    call_kwargs = mock_vector.upsert_memory.call_args.kwargs
    assert call_kwargs["memory_type"] == "pattern"
    assert call_kwargs["description"] == "test pattern"
