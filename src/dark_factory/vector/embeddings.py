"""Embedding service using OpenAI text-embedding-3-large."""

from __future__ import annotations

import openai

MAX_CHARS = 30_000  # ~8000 tokens safe limit for text-embedding-3-large


class EmbeddingService:
    """Generate embeddings via OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-large") -> None:
        self.model = model
        self._client = openai.OpenAI()

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Truncates to MAX_CHARS."""
        text = text[:MAX_CHARS]
        response = self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one call."""
        truncated = [t[:MAX_CHARS] for t in texts]
        response = self._client.embeddings.create(
            model=self.model,
            input=truncated,
        )
        return [item.embedding for item in response.data]
