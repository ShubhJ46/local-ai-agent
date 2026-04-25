from app.embed import get_embedding
from app.vector_store import client
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import uuid


MEMORY_COLLECTION = "memory"

class ShortTermMemory:
    def __init__(self, max_messages=5):
        self.history = []
        self.max_messages = max_messages

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        
        if len(self.history) > self.max_messages:
            self.history.pop(0)

    def get_context(self):
        return "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self.history]
        )
    
def init_memory_collection(vector_size):
    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if MEMORY_COLLECTION not in names:
        client.create_collection(
            collection_name=MEMORY_COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

def store_memory(text):
    embedding = get_embedding(text)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"text": text}
    )

    client.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[point]
    )

def retrieve_memory(query, top_k=3):
    try:
        query_embedding = get_embedding(query)

        results = client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=query_embedding,
            limit=top_k,
            query_filter=...   # advanced (we can add later)
        ).points

        return [r.payload["text"] for r in results]

    except Exception:
        return []
