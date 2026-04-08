"""Vector repository: upsert and search embeddings in Qdrant."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from dark_factory.vector.client import QdrantClientWrapper
from dark_factory.vector.embeddings import EmbeddingService

if TYPE_CHECKING:
    from dark_factory.models.domain import CodeArtifact, Spec

log = structlog.get_logger()


class VectorRepository:
    """Semantic search over memories, specs, and code artifacts."""

    def __init__(self, client: QdrantClientWrapper, embeddings: EmbeddingService) -> None:
        self._client = client
        self._embeddings = embeddings

    # ── Memory operations ────────────────────────────────────────────

    def upsert_memory(
        self,
        *,
        node_id: str,
        memory_type: str,
        description: str,
        secondary_text: str,
        source_feature: str,
        source_spec_id: str = "",
        agent: str = "",
        relevance_score: float = 0.5,
    ) -> None:
        text = f"{description}\n{secondary_text}"
        vector = self._embeddings.embed(text)
        self._client.client.upsert(
            collection_name=self._client.collection_name("memories"),
            points=[
                PointStruct(
                    id=self._to_point_id(node_id),
                    vector=vector,
                    payload={
                        "id": node_id,
                        "memory_type": memory_type,
                        "description": description,
                        "secondary_text": secondary_text,
                        "source_feature": source_feature,
                        "source_spec_id": source_spec_id,
                        "agent": agent,
                        "relevance_score": relevance_score,
                    },
                )
            ],
        )

    def search_memories(
        self,
        *,
        query_text: str,
        memory_type: str | None = None,
        source_feature: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        vector = self._embeddings.embed(query_text)
        conditions = []
        if memory_type:
            conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))
        if source_feature:
            conditions.append(FieldCondition(key="source_feature", match=MatchValue(value=source_feature)))

        query_filter = Filter(must=conditions) if conditions else None
        results = self._client.client.query_points(
            collection_name=self._client.collection_name("memories"),
            query=vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=0.3,
        ).points
        return [{"id": p.payload.get("id", ""), "score": p.score, **p.payload} for p in results]

    # ── Spec operations ──────────────────────────────────────────────

    def upsert_spec(self, *, spec: Spec) -> None:
        criteria = "\n".join(spec.acceptance_criteria)
        text = f"{spec.title}\n{spec.description}\n{criteria}"
        vector = self._embeddings.embed(text)
        self._client.client.upsert(
            collection_name=self._client.collection_name("specs"),
            points=[
                PointStruct(
                    id=self._to_point_id(spec.id),
                    vector=vector,
                    payload={
                        "id": spec.id,
                        "title": spec.title,
                        "description": spec.description,
                        "capability": spec.capability,
                        "acceptance_criteria": criteria,
                        "requirement_ids": spec.requirement_ids,
                    },
                )
            ],
        )

    def search_similar_specs(self, *, query_text: str, limit: int = 5) -> list[dict]:
        vector = self._embeddings.embed(query_text)
        results = self._client.client.query_points(
            collection_name=self._client.collection_name("specs"),
            query=vector,
            limit=limit,
            score_threshold=0.3,
        ).points
        return [{"id": p.payload.get("id", ""), "score": p.score, **p.payload} for p in results]

    # ── Code operations ──────────────────────────────────────────────

    def upsert_code(self, *, artifact: CodeArtifact) -> None:
        text = f"{artifact.file_path}\n{artifact.content[:30000]}"
        vector = self._embeddings.embed(text)
        self._client.client.upsert(
            collection_name=self._client.collection_name("code"),
            points=[
                PointStruct(
                    id=self._to_point_id(artifact.id),
                    vector=vector,
                    payload={
                        "id": artifact.id,
                        "spec_id": artifact.spec_id,
                        "file_path": artifact.file_path,
                        "language": artifact.language,
                        "content_preview": artifact.content[:2000],
                    },
                )
            ],
        )

    def search_similar_code(
        self, *, query_text: str, language: str | None = None, limit: int = 5,
    ) -> list[dict]:
        vector = self._embeddings.embed(query_text)
        conditions = []
        if language:
            conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
        query_filter = Filter(must=conditions) if conditions else None
        results = self._client.client.query_points(
            collection_name=self._client.collection_name("code"),
            query=vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=0.3,
        ).points
        return [{"id": p.payload.get("id", ""), "score": p.score, **p.payload} for p in results]

    # ── Relevance sync ───────────────────────────────────────────────

    def update_relevance_score(self, *, node_id: str, new_score: float) -> None:
        self._client.client.set_payload(
            collection_name=self._client.collection_name("memories"),
            payload={"relevance_score": new_score},
            points=[self._to_point_id(node_id)],
        )

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_point_id(node_id: str) -> str:
        """Convert a Neo4j node ID to a Qdrant-compatible point ID.
        Qdrant accepts string IDs natively."""
        return node_id
