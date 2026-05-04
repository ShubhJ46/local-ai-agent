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

def rewrite_query(query):
    q = query.lower()

    if "getmapping" in q or "get mappings" in q:
        return query + " GET endpoint @GetMapping controller settlements"
    
    if "postmapping" in q or "post mappings" in q:
        return query + " POST endpoint controller mapping"
    
    if "settlement" in q:
        return query + " settlements endpoint controller GET"

    return query

def rerank(results, query):
    query = query.lower()

    def score(x):
        score = 0

        endpoint = (x.get("endpoint") or "").lower()
        name = (x.get("name") or "").lower()

        if "settlement" in query and "settlement" in endpoint:
            score += 2

        if "get" in query and x.get("http_method") == "GET":
            score += 2

        if "settlement" in name:
            score += 1

        return score

    return sorted(results, key=score, reverse=True)

def find_endpoint(path_query):
    query_embedding = get_embedding(rewrite_query(path_query))

    results = search(
        query_embedding,
        top_k=5,
        filter={"type": "endpoint"}
    )

    results = rerank(results, path_query)

    return results  

TOOLS = {
    "search_documents": search_documents,
    "read_file": read_file,
    "find_endpoint": find_endpoint
}