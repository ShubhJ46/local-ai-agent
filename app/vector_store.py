from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(path="./qdrant_data")

COLLECTION_NAME = "documents"

def init_collection(vector_size):
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

def store_embeddings(chunks):
    points = []

    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=i,
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

def close_client():
    client.close()

def     search(query_embedding, top_k=5, filter=None):
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k * 3
    ).points

    formatted = []

    for hit in results:
        payload = hit.payload
        metadata = payload.get("metadata", {})

        if metadata.get("type") == "endpoint":
            if not metadata.get("endpoint"):
                continue

        # Optional filtering 
        if filter:
            skip = False
            for k, v in filter.items():
                if metadata.get(k) != v:
                    skip = True
                    break
            if skip:
                continue

        item = {
            "text": payload.get("text"),
            "file": metadata.get("file_name"),
            "type": metadata.get("type"),
            "name": metadata.get("name"),
            "path": metadata.get("path"),
            "annotations": metadata.get("annotations", []),
            "endpoint": metadata.get("endpoint"),
            "http_method": metadata.get("http_method"),
        }

        formatted.append(item)

    # Smart ranking
    def score(x):
        if x["type"] == "endpoint":
            return 0
        elif x["type"] in ("method", "function"):
            return 1
        elif x["type"] == "class":
            return 2
        else:
            return 3

    formatted.sort(key=score)

    return formatted[:top_k]