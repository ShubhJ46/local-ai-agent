from unittest.mock import Mock, patch

import pytest
import requests

from app.embed import get_embeddings
from app.errors import OllamaError
from app.ingest.ingest import file_hash, plan_chunks
from app.vector_store import point_id


def test_plan_chunks_marks_parts_only_when_a_unit_actually_splits():
    documents = [
        {"content": "def small(): pass", "metadata": {"type": "function", "path": "/a.py"}},
        {"content": "x" * 50, "metadata": {"type": "file", "path": "/b.txt"}},
    ]

    planned = plan_chunks(documents)

    assert all("part" not in metadata for _text, metadata in planned)


def test_plan_chunks_gives_each_chunk_its_own_metadata():
    """Regression: one dict was shared, so a per-chunk field leaked to siblings."""
    long_unit = "".join(f"    step_{index}()\n" for index in range(400))
    planned = plan_chunks([{"content": long_unit, "metadata": {"type": "function"}}])

    assert len(planned) > 1
    first, second = planned[0][1], planned[1][1]
    first["mutated"] = True
    assert "mutated" not in second


def test_point_id_is_stable_and_distinguishes_chunks():
    metadata = {"path": "/src/A.java", "type": "method", "name": "save", "part": 1}

    assert point_id(metadata) == point_id(dict(metadata))
    assert point_id(metadata) != point_id({**metadata, "part": 2})
    assert point_id(metadata) != point_id({**metadata, "name": "load"})


def test_file_hash_changes_with_content():
    assert file_hash("a") == file_hash("a")
    assert file_hash("a") != file_hash("b")


@patch("app.embed.requests.post")
def test_get_embeddings_sends_one_request_for_many_texts(post):
    response = Mock(status_code=200)
    response.json.return_value = {"embeddings": [[0.1], [0.2], [0.3]]}
    post.return_value = response

    assert get_embeddings(["a", "b", "c"]) == [[0.1], [0.2], [0.3]]
    assert post.call_count == 1


@patch("app.embed.get_embedding", side_effect=lambda text: [len(text)])
@patch("app.embed.requests.post")
def test_get_embeddings_falls_back_when_the_batch_endpoint_is_absent(post, _single):
    """Older Ollama builds predate /api/embed; ingest must still work."""
    post.return_value = Mock(status_code=404)

    assert get_embeddings(["a", "bb"]) == [[1], [2]]


@patch("app.embed.requests.post")
def test_get_embeddings_rejects_a_truncated_response(post):
    response = Mock(status_code=200)
    response.json.return_value = {"embeddings": [[0.1]]}
    post.return_value = response

    with pytest.raises(OllamaError, match="1 embeddings for 2 inputs"):
        get_embeddings(["a", "b"])


@patch("app.embed.requests.post", side_effect=requests.Timeout())
def test_get_embeddings_wraps_network_failures(_post):
    with pytest.raises(OllamaError, match="Could not get embeddings"):
        get_embeddings(["a"])


def test_get_embeddings_handles_an_empty_batch():
    assert get_embeddings([]) == []


def _document(path, digest, name):
    return {
        "content": f"def {name}(): pass",
        "metadata": {"path": path, "file_hash": digest, "type": "function", "name": name},
    }


@pytest.fixture
def indexing(monkeypatch):
    """Run ingest_codebase against fakes, recording what it decided to do."""
    calls = {"stored": [], "deleted": set()}

    monkeypatch.setattr("app.ingest.ingest.get_embeddings", lambda texts: [[0.1]] * len(texts))
    monkeypatch.setattr("app.ingest.ingest.init_collection", lambda *a, **k: None)
    monkeypatch.setattr("app.ingest.ingest.reset_lexical_index", lambda: None)
    monkeypatch.setattr(
        "app.ingest.ingest.store_embeddings", lambda chunks: calls["stored"].extend(chunks)
    )
    monkeypatch.setattr(
        "app.ingest.ingest.delete_files", lambda paths: calls["deleted"].update(paths)
    )
    return calls


def test_ingest_skips_files_whose_content_is_unchanged(indexing, monkeypatch):
    documents = [_document("/a.py", "hash-a", "alpha"), _document("/b.py", "hash-b", "beta")]
    monkeypatch.setattr("app.ingest.ingest.load_documents", lambda _path: documents)
    monkeypatch.setattr(
        "app.ingest.ingest.indexed_file_hashes", lambda: {"/a.py": "hash-a", "/b.py": "hash-b"}
    )

    from app.ingest.ingest import ingest_codebase

    summary = ingest_codebase("/repo")

    assert summary == {"indexed": 0, "files_changed": 0, "files_removed": 0, "files_total": 2}
    assert indexing["stored"] == []


def test_ingest_re_embeds_only_the_changed_file(indexing, monkeypatch):
    documents = [_document("/a.py", "hash-a", "alpha"), _document("/b.py", "NEW", "beta")]
    monkeypatch.setattr("app.ingest.ingest.load_documents", lambda _path: documents)
    monkeypatch.setattr(
        "app.ingest.ingest.indexed_file_hashes", lambda: {"/a.py": "hash-a", "/b.py": "hash-b"}
    )

    from app.ingest.ingest import ingest_codebase

    summary = ingest_codebase("/repo")

    assert summary["files_changed"] == 1
    assert [chunk["metadata"]["path"] for chunk in indexing["stored"]] == ["/b.py"]
    # Old points for the changed file must go, or stale chunks survive.
    assert indexing["deleted"] == {"/b.py"}


def test_ingest_removes_points_for_deleted_files(indexing, monkeypatch):
    monkeypatch.setattr(
        "app.ingest.ingest.load_documents", lambda _path: [_document("/a.py", "hash-a", "alpha")]
    )
    monkeypatch.setattr(
        "app.ingest.ingest.indexed_file_hashes",
        lambda: {"/a.py": "hash-a", "/gone.py": "hash-gone"},
    )

    from app.ingest.ingest import ingest_codebase

    summary = ingest_codebase("/repo")

    assert summary["files_removed"] == 1
    assert indexing["deleted"] == {"/gone.py"}
