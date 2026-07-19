import hashlib
from collections.abc import Callable

from app.chunker import chunk_code_unit, chunk_text
from app.embed import get_embeddings
from app.loader import load_documents
from app.retrieval import reset_lexical_index
from app.vector_store import (
    delete_files,
    indexed_file_hashes,
    init_collection,
    store_embeddings,
)

# Documents the AST already reduced to a single meaningful unit. These are
# embedded whole; only prose and unparsed files are windowed.
CODE_UNIT_TYPES = frozenset({"class", "method", "endpoint", "function"})

# One request per chunk is dominated by round-trip overhead. Batching is worth
# roughly 3x; the size is a compromise between that and request payload size.
EMBED_BATCH_SIZE = 64

Progress = Callable[[int, int], None] | None


def file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def split_document(document: dict) -> list[str]:
    """Split a loaded document into the pieces that will each be embedded."""
    content = document["content"]
    if document["metadata"].get("type") in CODE_UNIT_TYPES:
        return chunk_code_unit(content)
    return chunk_text(content)


def plan_chunks(documents: list[dict]) -> list[tuple[str, dict]]:
    """Expand documents into the (text, metadata) pairs that will be embedded."""
    planned = []
    for document in documents:
        pieces = split_document(document)
        for index, piece in enumerate(pieces):
            # Copy per chunk: sharing one dict across chunks meant any per-chunk
            # field written later would leak across every sibling chunk.
            metadata = dict(document["metadata"])
            if len(pieces) > 1:
                metadata["part"] = index + 1
                metadata["part_count"] = len(pieces)
            planned.append((piece, metadata))
    return planned


def embed_chunks(planned: list[tuple[str, dict]], progress: Progress = None) -> list[dict]:
    chunks = []
    for start in range(0, len(planned), EMBED_BATCH_SIZE):
        batch = planned[start : start + EMBED_BATCH_SIZE]
        embeddings = get_embeddings([text for text, _metadata in batch])
        for (text, metadata), embedding in zip(batch, embeddings, strict=True):
            chunks.append({"text": text, "embedding": embedding, "metadata": metadata})
        if progress:
            progress(len(chunks), len(planned))
    return chunks


def ingest(folder_path, progress: Progress = None):
    """Embed every chunk of a codebase, ignoring what may already be indexed."""
    return embed_chunks(plan_chunks(load_documents(folder_path)), progress)


def ingest_codebase(folder_path, progress: Progress = None, full: bool = False) -> dict:
    """Index a codebase, re-embedding only what changed since the last run.

    Returns counts describing the work done, so the caller can report whether
    anything actually needed doing.
    """
    documents = load_documents(folder_path)

    # The loader hashes each file once; a document only carries the digest of
    # the file it came from.
    current_hashes = {
        document["metadata"]["path"]: document["metadata"]["file_hash"]
        for document in documents
        if document["metadata"].get("path") and document["metadata"].get("file_hash")
    }

    known_hashes = {} if full else indexed_file_hashes()

    changed_paths = {
        path for path, digest in current_hashes.items() if known_hashes.get(path) != digest
    }
    # Files indexed previously that are no longer present, or whose content
    # changed: their old points must go, or stale chunks survive re-indexing.
    removed_paths = set(known_hashes) - set(current_hashes)

    planned = plan_chunks(
        [document for document in documents if document["metadata"].get("path") in changed_paths]
    )
    chunks = embed_chunks(planned, progress)

    if chunks:
        init_collection(len(chunks[0]["embedding"]), reset=full)
    if changed_paths or removed_paths:
        if not full:
            delete_files(changed_paths | removed_paths)
        if chunks:
            store_embeddings(chunks)
        reset_lexical_index()

    return {
        "indexed": len(chunks),
        "files_changed": len(changed_paths),
        "files_removed": len(removed_paths),
        "files_total": len(current_hashes),
    }
