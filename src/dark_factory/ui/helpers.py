"""Shared helpers for the UI layer (framework-agnostic)."""

from __future__ import annotations

import os
from typing import Any

import structlog

from dark_factory.config import Settings, load_settings

log = structlog.get_logger()

_cached_settings: Settings | None = None


def get_settings() -> Settings:
    """Load and cache settings."""
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = load_settings()
    return _cached_settings


def build_llm(settings: Settings, model_override: str | None = None) -> Any:
    """Instantiate the configured LLM client.

    Parameters
    ----------
    model_override:
        If given, use this model id instead of ``settings.llm.model``.
        Used by multi-model routing for per-stage model selection.
    """
    model_id = model_override or settings.llm.model
    if settings.llm.provider == "anthropic":
        from dark_factory.llm.anthropic import AnthropicClient

        return AnthropicClient(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=model_id,
        )
    elif settings.llm.provider == "langchain":
        from dark_factory.llm.langchain import LangChainClient

        return LangChainClient(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=model_id,
        )
    elif settings.llm.provider == "claude-agent":
        from dark_factory.llm.claude_code import ClaudeAgentClient

        return ClaudeAgentClient(model=model_id)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm.provider}")


def get_neo4j_client(settings: Settings):
    """Create a Neo4j client (caller manages lifecycle)."""
    from dark_factory.graph.client import Neo4jClient

    return Neo4jClient(settings.neo4j)


def get_memory_repo(settings: Settings):
    """Create a memory repository if memory is enabled.

    Returns a 2-tuple (repo, client) when memory is enabled, else ``None``.
    Caller must close ``client`` when done.
    """
    if not settings.memory.enabled:
        return None
    from dark_factory.config import Neo4jConfig
    from dark_factory.graph.client import Neo4jClient
    from dark_factory.memory.repository import MemoryRepository
    from dark_factory.memory.schema import init_memory_schema

    mem_config = Neo4jConfig(
        uri=settings.neo4j.uri,
        database=settings.memory.database,
        user=settings.neo4j.user,
        password=settings.neo4j.password,
    )
    client = Neo4jClient(mem_config)
    init_memory_schema(client)
    return MemoryRepository(client), client
