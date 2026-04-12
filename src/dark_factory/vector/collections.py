"""Create and manage Qdrant collections."""

from __future__ import annotations

import structlog
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from dark_factory.vector.client import QdrantClientWrapper

log = structlog.get_logger()

DIMENSION = 3072  # text-embedding-3-large output dimension

# Collection suffix → payload index definitions
COLLECTION_INDEXES: dict[str, dict[str, PayloadSchemaType]] = {
    "memories": {
        "memory_type": PayloadSchemaType.KEYWORD,
        "source_feature": PayloadSchemaType.KEYWORD,
        "source_spec_id": PayloadSchemaType.KEYWORD,
        "agent": PayloadSchemaType.KEYWORD,
    },
    "specs": {
        "capability": PayloadSchemaType.KEYWORD,
    },
    "code": {
        "spec_id": PayloadSchemaType.KEYWORD,
        "language": PayloadSchemaType.KEYWORD,
    },
    # Phase 5 Stage 3: episodic memory. One Qdrant point per Episode
    # whose vector is the embedding of the LLM-generated narrative
    # summary. Payload indexes on feature + outcome + run_id power
    # the filtered retrieval paths in ``recall_episodes`` (planner
    # usually wants to filter by current feature).
    "episodes": {
        "feature": PayloadSchemaType.KEYWORD,
        "outcome": PayloadSchemaType.KEYWORD,
        "run_id": PayloadSchemaType.KEYWORD,
    },
}


def ensure_collections(wrapper: QdrantClientWrapper, dimension: int = DIMENSION) -> None:
    """Create collections if they don't exist and set up payload indexes."""
    client = wrapper.client

    for suffix, indexes in COLLECTION_INDEXES.items():
        name = wrapper.collection_name(suffix)

        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

        for field_name, field_type in indexes.items():
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as exc:
                # M13 fix: narrow the silent catch. "already exists" is the
                # expected idempotent-retry case; everything else (auth,
                # schema mismatch, network) should surface as a warning so
                # it's not masked the way the original bare `except: pass`
                # did. We can't import a specific Qdrant exception class
                # portably across versions, so we pattern-match the message.
                msg = str(exc).lower()
                if "already exist" in msg or "exists" in msg:
                    continue
                log.warning(
                    "qdrant_payload_index_failed",
                    collection=name,
                    field=field_name,
                    error=str(exc),
                )
        log.info("qdrant_collection_ensured", collection=name)
