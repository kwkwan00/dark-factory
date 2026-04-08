"""Create and manage Qdrant collections."""

from __future__ import annotations

from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from dark_factory.vector.client import QdrantClientWrapper

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
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=field_type,
            )
