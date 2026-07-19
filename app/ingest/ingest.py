from app.chunker import chunk_text
from app.embed import get_embedding
from app.loader import load_documents
from app.vector_store import init_collection, store_embeddings


def ingest_codebase(folder_path):
    data = ingest(folder_path)

    if data:
        vector_size = len(data[0]["embedding"])
        init_collection(vector_size)
        store_embeddings(data)

    return len(data)


def ingest(folder_path):
    docs = load_documents(folder_path)

    all_chunks = []

    for doc in docs:
        chunks = chunk_text(doc["content"])

        for chunk in chunks:
            embedding = get_embedding(chunk)

            all_chunks.append({"text": chunk, "embedding": embedding, "metadata": doc["metadata"]})

    return all_chunks
