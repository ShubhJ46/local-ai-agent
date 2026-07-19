from app.embed import get_embedding
from app.llm import query_llm
from app.memory import ShortTermMemory
from app.tools import rerank, rewrite_query
from app.vector_store import search

memory = ShortTermMemory()


def format_context(results: list[dict]) -> str:
    """Convert retrieved documents into concise, source-bearing model context."""
    blocks = []
    for result in results:
        header = f"{result.get('type', '').upper()}: {result.get('name')}"
        if result.get("type") == "endpoint":
            header = f"ENDPOINT: {result.get('http_method')} {result.get('endpoint')}"
        source = (result.get("text") or "").strip()[:800]
        blocks.append(f"{header}\nFile: {result.get('file')}\nSource:\n{source}")
    return "\n\n---\n\n".join(blocks)


def agent_query(query: str, top_k: int = 3) -> str:
    """Answer a code question using only source retrieved from the active index."""
    rewritten_query = rewrite_query(query)
    query_embedding = get_embedding(rewritten_query)
    results = search(query_embedding, top_k=top_k)
    if not results:
        return "No indexed documents were found. Run `load <path>` first."

    context = format_context(rerank(results, query))
    prompt = f"""
Answer using only the retrieved source. Use at most 120 words.
State where the behavior is implemented, how it works, and cite file names.
If a mechanism is missing, say it is not implemented. Do not invent details.

RETRIEVED SOURCE:
{context}

QUESTION:
{query}
"""
    answer = query_llm(prompt).removeprefix("FINAL_ANSWER:").strip()
    memory.add("User", query)
    memory.add("AI", answer)
    return answer
