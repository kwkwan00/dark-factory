"""Embedding service using OpenAI text-embedding-3-large."""

from __future__ import annotations

import openai
import structlog

log = structlog.get_logger()

MAX_CHARS = 30_000  # ~8000 tokens safe limit for text-embedding-3-large


class EmbeddingService:
    """Generate embeddings via OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-large", api_key: str | None = None) -> None:
        self.model = model
        import os

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            log.warning("openai_api_key_missing", hint="Set OPENAI_API_KEY or pass api_key to EmbeddingService")
        self._client = openai.OpenAI(api_key=key)

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Truncates to MAX_CHARS."""
        log.debug("embedding", model=self.model, text_len=len(text))
        text = text[:MAX_CHARS]
        response = self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, chunked to respect API batch limits."""
        truncated = [t[:MAX_CHARS] for t in texts]
        batch_size = 2048
        all_embeddings: list[list[float]] = []
        for i in range(0, len(truncated), batch_size):
            chunk = truncated[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=chunk,
            )
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings
