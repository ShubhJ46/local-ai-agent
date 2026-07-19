import pytest

from app.chunker import chunk_code_unit, chunk_text
from app.ingest.ingest import split_document


def test_chunk_text_creates_overlapping_windows():
    assert chunk_text("abcdefgh", chunk_size=4, overlap=1) == ["abcd", "defg", "gh"]


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (4, -1), (4, 4)])
def test_chunk_text_rejects_invalid_sizes(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=chunk_size, overlap=overlap)


def test_chunk_code_unit_keeps_a_unit_that_fits_intact():
    function = "def useful():\n    return 42\n"
    assert chunk_code_unit(function, max_chars=100) == [function]


def test_chunk_code_unit_splits_only_on_line_boundaries():
    body = "".join(f"    line_{index}()\n" for index in range(10))

    parts = chunk_code_unit(body, max_chars=60)

    assert len(parts) > 1
    assert "".join(parts) == body
    for part in parts:
        assert part.endswith("\n")
        assert not part.startswith(" " * 4 + "line_") or part.lstrip().startswith("line_")


def test_chunk_code_unit_emits_an_overlong_line_whole():
    single_line = "x" * 500 + "\n"
    assert chunk_code_unit(single_line, max_chars=100) == [single_line]


def test_split_document_routes_code_units_away_from_character_windows():
    """An AST unit must not be windowed just because it is long."""
    long_function = "".join(f"    step_{index}()\n" for index in range(300))

    code_unit = split_document({"content": long_function, "metadata": {"type": "function"}})
    prose = split_document({"content": long_function, "metadata": {"type": "file"}})

    assert len(code_unit) < len(prose)
    assert all(part.endswith("\n") for part in code_unit)
