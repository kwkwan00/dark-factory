"""Vector repository: upsert and search embeddings in Qdrant."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from dark_factory.log import trace_methods
from dark_factory.vector.client import QdrantClientWrapper
from dark_factory.vector.embeddings import EmbeddingService

if TYPE_CHECKING:
    from dark_factory.models.domain import CodeArtifact, Spec

log = structlog.get_logger()

# Fixed namespace for converting Neo4j string IDs → Qdrant point UUIDs.
# Qdrant only accepts unsigned int or UUID as point ID; we use uuid5 with
# this namespace so the same node ID always maps to the same point ID
# (deterministic, reversible-by-name, no extra storage needed).
_QDRANT_ID_NAMESPACE = uuid.UUID("9c8d7e6f-5a4b-3c2d-1e0f-aabbccddeeff")


@trace_methods
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
        run_id: str = "",
    ) -> None:
        text = f"{description}\n{secondary_text}"
        vector = self._embeddings.embed(text)
        payload: dict[str, Any] = {
            "id": node_id,
            "memory_type": memory_type,
            "description": description,
            "secondary_text": secondary_text,
            "source_feature": source_feature,
            "source_spec_id": source_spec_id,
            "agent": agent,
            "relevance_score": relevance_score,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "times_recalled": 0,
        }
        self._client.client.upsert(
            collection_name=self._client.collection_name("memories"),
            points=[
                PointStruct(
                    id=self._to_point_id(node_id),
                    vector=vector,
                    payload=payload,
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

    # ── Refinery v2: role-filtered hybrid retrieval ──────────────────
    #
    # search_memories_hybrid carries a RoleFilterPolicy — required, no
    # no-filter overload — so a policy is always translated into a
    # server-side Qdrant Filter before the query runs. This is the
    # Parnas-discipline load-bearing method: an out-of-slice row cannot
    # be returned even if the calling role's prompt asks for one.
    #
    # Phase 6 ships dense-only server-side filtering. The Phase-11
    # sparse/BM25 migration adds a ``bm25`` named vector to the
    # ``memories`` collection; at that point this method switches to
    # ``prefetch`` + ``FusionQuery(RRF)`` without the caller noticing.

    def search_memories_hybrid(
        self,
        *,
        query_text: str,
        policy: "Any",  # RoleFilterPolicy — avoid circular import
        limit: int | None = None,
    ) -> list[dict]:
        """Dense + (future) BM25 search with role-filter enforcement.

        Mandatory ``policy`` argument — there is no no-filter overload.
        Callers pass ``role_slices.ROLE_FILTERS[role]`` or an override;
        the filter is built in one place (``context.translators``) and
        passed as ``query_filter`` so Qdrant enforces the slice.
        """

        from dark_factory.api.refinery.context.translators import (
            policy_to_qdrant_memory_filter,
        )

        qdrant_filter = policy_to_qdrant_memory_filter(policy)
        effective_limit = limit or getattr(policy, "row_cap", 12) or 12
        vector = self._embeddings.embed(query_text)

        results = self._client.client.query_points(
            collection_name=self._client.collection_name("memories"),
            query=vector,
            query_filter=qdrant_filter,  # NEVER None-unless-policy-is-empty
            limit=effective_limit,
            score_threshold=0.3,
        ).points
        return [
            {"id": p.payload.get("id", ""), "score": p.score, **p.payload}
            for p in results
        ]

    # ── Spec operations ──────────────────────────────────────────────

    def upsert_spec(
        self,
        *,
        spec: Spec,
        eval_score: float | None = None,
        attempts: int | None = None,
    ) -> None:
        criteria = "\n".join(spec.acceptance_criteria)
        scenarios_text = "\n".join(
            f"WHEN {s.when} THEN {s.then}" for s in spec.scenarios
        )
        text = f"{spec.title}\n{spec.description}\n{criteria}\n{scenarios_text}"
        vector = self._embeddings.embed(text)
        payload: dict[str, Any] = {
            "id": spec.id,
            "title": spec.title,
            "description": spec.description,
            "capability": spec.capability,
            "acceptance_criteria": spec.acceptance_criteria,
            "requirement_ids": spec.requirement_ids,
            "dependencies": spec.dependencies,
            "scenarios": [s.model_dump() for s in spec.scenarios],
        }
        if eval_score is not None:
            payload["eval_score"] = eval_score
        if attempts is not None:
            payload["attempts"] = attempts
        self._client.client.upsert(
            collection_name=self._client.collection_name("specs"),
            points=[
                PointStruct(
                    id=self._to_point_id(spec.id),
                    vector=vector,
                    payload=payload,
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

    # ── Episode operations ───────────────────────────────────────────
    #
    # Episodic memory stores the autobiographical narrative of one
    # feature swarm run. The caller (``EpisodeWriter``) is
    # responsible for producing the embedding vector — we accept it
    # as a pre-computed argument rather than re-embedding here, so
    # the writer's error handling can surface embedding failures
    # distinctly from Qdrant-side failures.

    def upsert_episode(
        self,
        *,
        episode_id: str,
        run_id: str,
        feature: str,
        outcome: str,
        summary: str,
        vector: list[float],
        turns_used: int = 0,
        duration_seconds: float = 0.0,
        spec_ids: list[str] | None = None,
        final_eval_scores: dict[str, float] | None = None,
        recalled_memory_ids: list[str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "id": episode_id,
            "run_id": run_id,
            "feature": feature,
            "outcome": outcome,
            "summary": summary,
            "turns_used": turns_used,
            "duration_seconds": duration_seconds,
        }
        if spec_ids:
            payload["spec_ids"] = spec_ids
        if final_eval_scores:
            payload["final_eval_scores"] = final_eval_scores
        if recalled_memory_ids:
            payload["recalled_memory_ids"] = recalled_memory_ids
        self._client.client.upsert(
            collection_name=self._client.collection_name("episodes"),
            points=[
                PointStruct(
                    id=self._to_point_id(episode_id),
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def search_episodes(
        self,
        *,
        query_text: str,
        feature: str | None = None,
        outcome: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Semantic search over episode summaries.

        Used as the Qdrant half of the ``recall_episodes`` hybrid
        RRF merge. Returns the same ``{id, score, **payload}`` shape
        as ``search_memories`` so the merge helper can treat them
        identically.
        """
        vector = self._embeddings.embed(query_text)
        conditions = []
        if feature:
            conditions.append(
                FieldCondition(key="feature", match=MatchValue(value=feature))
            )
        if outcome and outcome.lower() != "any":
            conditions.append(
                FieldCondition(key="outcome", match=MatchValue(value=outcome.lower()))
            )
        query_filter = Filter(must=conditions) if conditions else None
        results = self._client.client.query_points(
            collection_name=self._client.collection_name("episodes"),
            query=vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=0.3,
        ).points
        return [
            {"id": p.payload.get("id", ""), "score": p.score, **p.payload}
            for p in results
        ]

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_point_id(node_id: str) -> str:
        """Convert a string node ID to a Qdrant-compatible point UUID.

        Qdrant only accepts unsigned int or UUID as point IDs (it rejects
        free-form strings like "mistake-b6d011d5"). We generate a
        deterministic UUID5 from the node ID, so:

        - The same node ID always maps to the same point ID (idempotent
          upserts).
        - The original node ID is preserved in the payload's ``id`` field
          for retrieval/display.
        """
        return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, node_id))
