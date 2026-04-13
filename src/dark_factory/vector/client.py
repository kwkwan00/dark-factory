"""Qdrant client wrapper following the same pattern as graph/client.py."""

from __future__ import annotations

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from dark_factory.config import QdrantConfig

log = structlog.get_logger()


class QdrantClientWrapper:
    """Wraps the Qdrant SDK client with config and lifecycle management."""

    def __init__(self, config: QdrantConfig) -> None:
        self.config = config
        kwargs: dict = {"url": config.url}
        api_key_val = config.api_key.get_secret_value() if config.api_key else ""
        if api_key_val:
            kwargs["api_key"] = api_key_val
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
        except (ConnectionError, TimeoutError, UnexpectedResponse, ResponseHandlingException, OSError):
            return False

    def close(self) -> None:
        self._client.close()
