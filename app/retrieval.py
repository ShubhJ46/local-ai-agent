"""Hybrid retrieval: dense vector search fused with BM25, then type-prioritised.

Embeddings and BM25 fail differently. Embeddings handle paraphrase ("how do I
add a pet") but drift on exact identifiers; BM25 nails identifiers
(``OwnerRepository``) but cannot bridge vocabulary. Fusing their rankings
recovers results that neither retriever ranks highly on its own.

Reciprocal rank fusion is used rather than score blending because the two
retrievers produce incomparable scales — cosine similarity and BM25 relevance
cannot be added meaningfully, but their *ranks* can.
"""

from app.embed import get_embedding
from app.lexical import BM25Index
from app.vector_store import iter_points, search

# Standard RRF damping. Large enough that the top few ranks are close together,
# so a result both retrievers like beats one that either ranks first alone.
RRF_K = 60

# A question about behaviour is usually answered by a handler, not by the class
# that contains it or the file that contains that. Applied after fusion so it
# breaks ties without overriding a strong agreement between retrievers.
TYPE_PRIORITY = {
    "endpoint": 3,
    "method": 2,
    "function": 2,
    "class": 1,
    "file": 0,
}
TYPE_PRIORITY_WEIGHT = 0.002

_lexical_index: BM25Index | None = None
_lexical_documents: dict[object, dict] = {}


def reset_lexical_index() -> None:
    """Drop the cached lexical index. Call after the collection changes."""
    global _lexical_index
    _lexical_index = None
    _lexical_documents.clear()


def get_lexical_index() -> BM25Index:
    """Build the BM25 index from the stored chunks, once per index generation.

    The whole corpus is held in memory. That is fine at the scale this tool
    targets and keeps the lexical half dependency-free; a larger corpus would
    want a real inverted index on disk.
    """
    global _lexical_index
    if _lexical_index is None:
        points = iter_points()
        _lexical_documents.clear()
        for point in points:
            _lexical_documents[point["id"]] = point
        _lexical_index = BM25Index(
            [(point["id"], _searchable_text(point)) for point in points]
        )
    return _lexical_index


def _searchable_text(point: dict) -> str:
    """Include the symbol and route in the lexical text, not just the body.

    A query naming an endpoint path should match the chunk serving it even when
    the path appears only in metadata.
    """
    parts = [
        point.get("name") or "",
        point.get("endpoint") or "",
        point.get("http_method") or "",
        point.get("file") or "",
        point.get("text") or "",
    ]
    return "\n".join(part for part in parts if part)


def type_priority(result: dict) -> int:
    return TYPE_PRIORITY.get(result.get("type"), 0)


def fuse(rankings: list[list[dict]], top_k: int) -> list[dict]:
    """Combine ranked result lists with reciprocal rank fusion."""
    scores: dict[object, float] = {}
    documents: dict[object, dict] = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            identifier = result["id"]
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (RRF_K + rank)
            documents.setdefault(identifier, result)

    ranked = sorted(
        documents.values(),
        key=lambda result: (
            scores[result["id"]] + TYPE_PRIORITY_WEIGHT * type_priority(result)
        ),
        reverse=True,
    )
    return ranked[:top_k]


def lexical_search(query: str, top_k: int = 5) -> list[dict]:
    index = get_lexical_index()
    return [_lexical_documents[identifier] for identifier, _score in index.search(query, top_k)]


def vector_search(query: str, top_k: int = 5) -> list[dict]:
    return search(get_embedding(query), top_k=top_k)


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve with both retrievers and fuse the rankings.

    Each retriever is asked for more candidates than are returned, so a result
    ranked modestly by both can still surface above one ranked highly by only
    a single retriever.
    """
    candidate_k = max(top_k * 2, top_k)
    return fuse(
        [vector_search(query, candidate_k), lexical_search(query, candidate_k)],
        top_k=top_k,
    )
