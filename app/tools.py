"""Tools the agent can call while answering a question."""

from pathlib import Path

from app.embed import get_embedding
from app.retrieval import hybrid_search
from app.vector_store import iter_points
from app.vector_store import search as vector_search_raw


def format_results(results: list[dict]) -> str:
    """Render results with their location so the model can cite them."""
    if not results:
        return "No matching code was found in the index."

    blocks = []
    for result in results:
        location = result.get("file") or "unknown file"
        if result.get("start_line") and result.get("end_line"):
            location = f"{location}:{result['start_line']}-{result['end_line']}"

        header = f"{(result.get('type') or 'code').upper()} {result.get('name') or ''}".strip()
        if result.get("type") == "endpoint" and result.get("endpoint"):
            header = f"ENDPOINT {result.get('http_method')} {result['endpoint']}"

        blocks.append(f"{header}\n{location}\n{(result.get('text') or '').strip()[:800]}")

    return "\n\n---\n\n".join(blocks)


def search_documents(query: str) -> str:
    """Search the indexed codebase for code relevant to a question."""
    return format_results(hybrid_search(query, top_k=3))


def find_endpoint(path_query: str) -> str:
    """Find the HTTP endpoint whose route or handler matches the query."""
    results = vector_search_raw(
        get_embedding(path_query), top_k=5, metadata_filter={"type": "endpoint"}
    )
    return format_results(results)


def read_file(file_name: str) -> str:
    """Read an indexed source file by name.

    Resolved against the index rather than a fixed directory, so it works for
    whatever codebase is currently loaded.
    """
    matches = {
        point["path"]
        for point in iter_points()
        if point.get("file") == file_name or point.get("path") == file_name
    }

    if not matches:
        return f"No indexed file named {file_name}."

    path = Path(sorted(matches)[0])
    if not path.is_file():
        return f"{path} is indexed but no longer on disk."

    return path.read_text(encoding="utf-8", errors="ignore")[:8_000]


TOOLS = {
    "search_documents": search_documents,
    "find_endpoint": find_endpoint,
    "read_file": read_file,
}

TOOL_DESCRIPTIONS = {
    "search_documents": "search_documents(query) - semantic + keyword search over the indexed code",
    "find_endpoint": "find_endpoint(path_or_description) - locate an HTTP endpoint handler",
    "read_file": "read_file(file_name) - read the full source of an indexed file",
}
