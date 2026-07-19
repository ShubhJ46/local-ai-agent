import uuid

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

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return the shared client, connecting on first use.

    Constructing this at import time took an exclusive lock on the index
    directory as a side effect of importing the module, which broke unrelated
    processes and made tests depend on whatever else happened to be running.
    """
    global _client
    if _client is None:
        if settings.qdrant_url:
            # Server mode: several processes can share one index, so the CLI and
            # the HTTP API can run at the same time.
            _client = QdrantClient(url=settings.qdrant_url)
        else:
            _client = QdrantClient(path=settings.vector_store_path)
    return _client


COLLECTION_NAME = "documents"


# Fixed namespace so a chunk keeps the same id across runs and machines.
POINT_NAMESPACE = uuid.UUID("6f9f3a52-1f0e-5c7a-9a1d-0d5b3f9c4e21")

UPSERT_BATCH_SIZE = 256


def point_id(metadata: dict) -> str:
    """Derive a stable id from what identifies a chunk.

    Positional ids were only safe while ingest wrote the whole corpus in one
    call. Writing in batches, or re-indexing a single changed file, would
    otherwise overwrite unrelated chunks or duplicate the same one.
    """
    key = "|".join(
        str(metadata.get(field) or "") for field in ("path", "type", "name", "part")
    )
    return str(uuid.uuid5(POINT_NAMESPACE, key))


def collection_exists() -> bool:
    collections = get_client().get_collections().collections
    return any(collection.name == COLLECTION_NAME for collection in collections)


def init_collection(vector_size: int, reset: bool = False) -> None:
    """Ensure the collection exists, keeping its contents unless asked otherwise."""
    if reset and collection_exists():
        get_client().delete_collection(collection_name=COLLECTION_NAME)

    if not collection_exists():
        get_client().create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def store_embeddings(chunks: list[dict]) -> None:
    points = [
        PointStruct(
            id=point_id(chunk["metadata"]),
            vector=chunk["embedding"],
            payload={"text": chunk["text"], "metadata": chunk["metadata"]},
        )
        for chunk in chunks
    ]

    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        get_client().upsert(
            collection_name=COLLECTION_NAME,
            points=points[start : start + UPSERT_BATCH_SIZE],
        )


def indexed_file_hashes() -> dict[str, str]:
    """Return the content hash recorded for each indexed file path.

    Used to decide what actually needs re-embedding. Files indexed before
    hashes were recorded simply compare unequal and are refreshed.
    """
    if not collection_exists():
        return {}

    hashes: dict[str, str] = {}
    offset = None
    while True:
        batch, offset = get_client().scroll(
            collection_name=COLLECTION_NAME,
            limit=1024,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            metadata = (point.payload or {}).get("metadata", {})
            path, digest = metadata.get("path"), metadata.get("file_hash")
            if path and digest:
                hashes[path] = digest
        if offset is None:
            break
    return hashes


def delete_files(paths: set[str]) -> None:
    """Remove every point belonging to the given files."""
    if not paths or not collection_exists():
        return

    get_client().delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            should=[
                FieldCondition(key="metadata.path", match=MatchValue(value=path))
                for path in sorted(paths)
            ]
        ),
    )


def close_client():
    """Close the connection and forget it, so a later call reconnects."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


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
        batch, offset = get_client().scroll(
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
    results = get_client().query_points(
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
