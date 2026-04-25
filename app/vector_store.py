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

def search(query_embedding, top_k=5):
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k
    ).points

    filtered = [
        hit.payload for hit in results
        if hit.payload["metadata"]["file_name"].endswith((".cpp", ".py"))
    ]

    return filtered[:top_k]