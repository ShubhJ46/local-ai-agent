from app.chunker import chunk_code_unit, chunk_text
from app.embed import get_embedding
from app.loader import load_documents
from app.retrieval import reset_lexical_index
from app.vector_store import init_collection, store_embeddings

# Documents the AST already reduced to a single meaningful unit. These are
# embedded whole; only prose and unparsed files are windowed.
CODE_UNIT_TYPES = frozenset({"class", "method", "endpoint", "function"})


def ingest_codebase(folder_path):
    data = ingest(folder_path)

    if data:
        vector_size = len(data[0]["embedding"])
        init_collection(vector_size)
        store_embeddings(data)
        # The lexical index is derived from the collection, so it is stale now.
        reset_lexical_index()

    return len(data)


def split_document(document: dict) -> list[str]:
    """Split a loaded document into the pieces that will each be embedded."""
    content = document["content"]
    if document["metadata"].get("type") in CODE_UNIT_TYPES:
        return chunk_code_unit(content)
    return chunk_text(content)


def ingest(folder_path):
    documents = load_documents(folder_path)

    all_chunks = []

    for document in documents:
        pieces = split_document(document)

        for index, piece in enumerate(pieces):
            # Copy per chunk: sharing one dict across chunks meant any per-chunk
            # field written later would leak across every sibling chunk.
            metadata = dict(document["metadata"])
            if len(pieces) > 1:
                metadata["part"] = index + 1
                metadata["part_count"] = len(pieces)

            all_chunks.append(
                {"text": piece, "embedding": get_embedding(piece), "metadata": metadata}
            )

    return all_chunks
