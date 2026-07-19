from unittest.mock import patch

from app.tools import TOOL_DESCRIPTIONS, TOOLS, format_results, read_file
from app.vector_store import format_point

ENDPOINT = {
    "id": 1,
    "type": "endpoint",
    "name": "processCreationForm",
    "file": "OwnerController.java",
    "path": "/repo/OwnerController.java",
    "text": "public String processCreationForm() {}",
    "http_method": "POST",
    "endpoint": "/owners/new",
    "start_line": 77,
    "end_line": 88,
}


def test_format_results_leads_with_the_route_for_an_endpoint():
    rendered = format_results([ENDPOINT])

    assert "ENDPOINT POST /owners/new" in rendered
    assert "OwnerController.java:77-88" in rendered


def test_format_results_falls_back_to_the_file_when_lines_are_unknown():
    rendered = format_results([{**ENDPOINT, "start_line": None, "end_line": None}])

    assert "OwnerController.java" in rendered
    assert ":77-88" not in rendered


def test_format_results_reports_an_empty_result_set():
    assert "No matching code" in format_results([])


def test_every_tool_is_described_for_the_model():
    """A tool the model is never told about can never be called."""
    assert set(TOOLS) == set(TOOL_DESCRIPTIONS)


def test_read_file_resolves_against_the_index(tmp_path):
    source = tmp_path / "Sample.java"
    source.write_text("class Sample {}")
    indexed = [{"file": "Sample.java", "path": str(source)}]

    with patch("app.tools.iter_points", return_value=indexed):
        assert "class Sample" in read_file("Sample.java")


def test_read_file_reports_a_name_that_was_never_indexed():
    with patch("app.tools.iter_points", return_value=[]):
        assert "No indexed file" in read_file("Missing.java")


def test_read_file_reports_a_file_that_vanished_from_disk(tmp_path):
    indexed = [{"file": "Gone.java", "path": str(tmp_path / "Gone.java")}]

    with patch("app.tools.iter_points", return_value=indexed):
        assert "no longer on disk" in read_file("Gone.java")


def test_format_point_hides_an_endpoint_with_no_route():
    """A handler annotated but without a path is not navigable; it must not surface."""
    payload = {"text": "code", "metadata": {"type": "endpoint", "endpoint": ""}}

    assert format_point(1, payload) is None


def test_format_point_exposes_line_range_and_identity():
    payload = {
        "text": "code",
        "metadata": {
            "type": "method",
            "name": "save",
            "file_name": "A.java",
            "start_line": 3,
            "end_line": 9,
        },
    }

    item = format_point("abc", payload)

    assert item["id"] == "abc"
    assert (item["start_line"], item["end_line"]) == (3, 9)
