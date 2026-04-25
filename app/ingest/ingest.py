from app.loader import load_documents
from app.chunker import chunk_text
from app.embed import get_embedding
from app.vector_store import init_collection, store_embeddings
from app.vector_store import search
from app.vector_store import close_client
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

            all_chunks.append({
                "text": chunk,
                "embedding": embedding,
                "metadata": doc["metadata"]
            })

    return all_chunks


if __name__ == "__main__":
    data = ingest("data")

    if len(data) > 0:
        vector_size = len(data[0]["embedding"])
        init_collection(vector_size)
        store_embeddings(data)

    print(f"Stored {len(data)} chunks")

    #TEST SEARCH HERE
    query = "main function"
    query_embedding = get_embedding(query)
    results = search(query_embedding)

    print("\nSearch Results:")
    for r in results:
        print("\n---")
        print(r["text"])

    close_client()