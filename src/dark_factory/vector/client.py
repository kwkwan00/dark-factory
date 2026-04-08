"""Qdrant client wrapper following the same pattern as graph/client.py."""

from __future__ import annotations

import structlog
from qdrant_client import QdrantClient

from dark_factory.config import QdrantConfig

log = structlog.get_logger()


class QdrantClientWrapper:
    """Wraps the Qdrant SDK client with config and lifecycle management."""

    def __init__(self, config: QdrantConfig) -> None:
        self.config = config
        kwargs: dict = {"url": config.url}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        self._client = QdrantClient(**kwargs)

    @property
    def client(self) -> QdrantClient:
        return self._client

    def collection_name(self, suffix: str) -> str:
        return f"{self.config.collection_prefix}_{suffix}"

    def is_available(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()
