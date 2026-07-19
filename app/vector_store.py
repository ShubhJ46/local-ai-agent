from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings

client = QdrantClient(path=settings.vector_store_path)

COLLECTION_NAME = "documents"


def init_collection(vector_size: int) -> None:
    """Create a fresh index. Re-indexing intentionally replaces prior contents."""
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def store_embeddings(chunks: list[dict]) -> None:
    points = []

    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=i,
                vector=chunk["embedding"],
                payload={"text": chunk["text"], "metadata": chunk["metadata"]},
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def close_client():
    client.close()


def format_point(point_id, payload: dict) -> dict | None:
    """Flatten a stored point into a retrieval result.

    Returns None for entries that should never surface, so the vector and
    lexical paths apply the same rule.
    """
    metadata = payload.get("metadata", {})

    if metadata.get("type") == "endpoint" and not metadata.get("endpoint"):
        return None

    return {
        # Stable within an index. Used as the join key when fusing rankings
        # from separate retrievers.
        "id": point_id,
        "text": payload.get("text"),
        "file": metadata.get("file_name"),
        "type": metadata.get("type"),
        "name": metadata.get("name"),
        "path": metadata.get("path"),
        "annotations": metadata.get("annotations", []),
        "endpoint": metadata.get("endpoint"),
        "http_method": metadata.get("http_method"),
        "start_line": metadata.get("start_line"),
        "end_line": metadata.get("end_line"),
    }


def iter_points() -> list[dict]:
    """Return every indexed chunk. Used to build the lexical index."""
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            item = format_point(point.id, point.payload or {})
            if item is not None:
                points.append(item)
        if offset is None:
            break
    return points


def search(
    query_embedding: list[float], top_k: int = 5, metadata_filter: dict | None = None
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_filter = None
    if metadata_filter:
        query_filter = Filter(
            must=[
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                for key, value in metadata_filter.items()
            ]
        )
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=query_filter,
    ).points

    formatted = []

    for hit in results:
        item = format_point(hit.id, hit.payload or {})
        if item is not None:
            formatted.append(item)

    return formatted
