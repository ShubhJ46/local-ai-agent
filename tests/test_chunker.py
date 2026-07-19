import pytest

from app.chunker import chunk_text


def test_chunk_text_creates_overlapping_windows():
    assert chunk_text("abcdefgh", chunk_size=4, overlap=1) == ["abcd", "defg", "gh"]


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (4, -1), (4, 4)])
def test_chunk_text_rejects_invalid_sizes(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=chunk_size, overlap=overlap)
