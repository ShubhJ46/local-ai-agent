from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.errors import OllamaError

RETRIEVED = [
    {
        "id": 1,
        "type": "endpoint",
        "name": "processCreationForm",
        "file": "OwnerController.java",
        "text": "public String processCreationForm() {}",
        "endpoint": "/owners/new",
        "http_method": "POST",
        "start_line": 77,
        "end_line": 88,
    }
]


@pytest.fixture
def client():
    return TestClient(app)


def test_query_returns_the_sources_the_answer_was_grounded_on(client):
    """Prose alone is unverifiable; the caller must be able to check the code."""
    with (
        patch("app.api.hybrid_search", return_value=RETRIEVED),
        patch("app.api.run_agent", return_value="Owners are created at /owners/new."),
    ):
        response = client.post("/query", json={"question": "How is an owner created?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Owners are created at /owners/new."
    assert body["sources"] == [
        {
            "file": "OwnerController.java",
            "lines": "77-88",
            "type": "endpoint",
            "name": "processCreationForm",
            "endpoint": "/owners/new",
            "http_method": "POST",
        }
    ]


def test_query_omits_a_line_range_that_is_unknown(client):
    partial = [{**RETRIEVED[0], "start_line": None, "end_line": None}]
    with (
        patch("app.api.hybrid_search", return_value=partial),
        patch("app.api.run_agent", return_value="An answer."),
    ):
        response = client.post("/query", json={"question": "How is an owner created?"})

    assert response.json()["sources"][0]["lines"] is None


def test_query_rejects_an_empty_question(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_reports_an_unreachable_model_as_unavailable(client):
    with (
        patch("app.api.hybrid_search", return_value=RETRIEVED),
        patch("app.api.run_agent", side_effect=OllamaError("Ollama is not running")),
    ):
        response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_index_returns_the_ingest_summary(client):
    summary = {"indexed": 118, "files_changed": 28, "files_removed": 0, "files_total": 28}
    with patch("app.api.ingest_codebase", return_value=summary):
        response = client.post("/index", json={"path": "/repo"})

    assert response.status_code == 200
    assert response.json() == summary


def test_index_rejects_a_path_that_is_not_a_directory(client):
    with patch("app.api.ingest_codebase", side_effect=ValueError("Index path is not a directory")):
        response = client.post("/index", json={"path": "/nope"})

    assert response.status_code == 400


def test_health_reports_how_much_is_indexed(client):
    with patch("app.api.iter_points", return_value=RETRIEVED):
        body = client.get("/health").json()

    assert body == {"status": "ok", "indexed_chunks": 1}


def test_health_is_still_usable_before_anything_is_indexed(client):
    with patch("app.api.iter_points", side_effect=ValueError("collection missing")):
        body = client.get("/health").json()

    assert body == {"status": "empty", "indexed_chunks": 0}
