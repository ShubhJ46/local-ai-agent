from app.embed import get_embedding
from app.vector_store import search

query = "authentication logic"

query_embedding = get_embedding(query)
results = search(query_embedding)

for r in results:
    print("\n---")
    print(r["text"])