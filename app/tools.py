from app.vector_store import search
from app.embed import get_embedding
import os

def search_documents(query):
    embedding = get_embedding(query)
    results = search(embedding, top_k=3)
    return "\n\n".join([r["text"] for r in results])


def read_file(file_name):
    base_path = "data"

    for root, _, files in os.walk(base_path):
        if file_name in files:
            path = os.path.join(root, file_name)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    return "File not found."


TOOLS = {
    "search_documents": search_documents,
    "read_file": read_file
}