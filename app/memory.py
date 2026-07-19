"""Two kinds of memory.

Short term is the current conversation, held in process, so a follow-up like
"explain that in more detail" has an antecedent. Long term is persisted in
Qdrant and recalled by similarity, so a question asked in an earlier session can
inform this one.
"""

import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.embed import get_embedding
from app.vector_store import client

MEMORY_COLLECTION = "memory"


class ShortTermMemory:
    def __init__(self, max_messages=6):
        self.history = []
        self.max_messages = max_messages

    def add(self, role, content):
        self.history.append({"role": role, "content": content})

        if len(self.history) > self.max_messages:
            self.history.pop(0)

    def get_context(self):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])

    def clear(self):
        self.history.clear()


def memory_collection_exists() -> bool:
    return any(
        collection.name == MEMORY_COLLECTION
        for collection in client.get_collections().collections
    )


def init_memory_collection(vector_size):
    """Create the long-term collection if it is missing. Safe to call repeatedly."""
    if not memory_collection_exists():
        client.create_collection(
            collection_name=MEMORY_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def store_memory(text):
    embedding = get_embedding(text)
    init_memory_collection(len(embedding))

    point = PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": text})

    client.upsert(collection_name=MEMORY_COLLECTION, points=[point])


def retrieve_memory(query, top_k=3):
    """Recall prior exchanges similar to the query.

    Returns nothing when no memory has been written yet. Failures to embed are
    left to propagate: if Ollama is unreachable the caller cannot answer either,
    and silently returning [] here previously hid a real defect for the entire
    life of this function.
    """
    if not memory_collection_exists():
        return []

    results = client.query_points(
        collection_name=MEMORY_COLLECTION,
        query=get_embedding(query),
        limit=top_k,
    ).points

    return [point.payload["text"] for point in results if point.payload]
