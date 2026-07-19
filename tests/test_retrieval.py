from unittest.mock import patch

import pytest

from app.retrieval import (
    fuse,
    get_lexical_index,
    hybrid_search,
    lexical_search,
    reset_lexical_index,
    type_priority,
)

INDEXED = [
    {"id": 1, "type": "endpoint", "name": "processCreationForm", "endpoint": "/owners/new",
     "http_method": "POST", "file": "OwnerController.java", "text": "save the owner"},
    {"id": 2, "type": "class", "name": "OwnerRepository", "endpoint": None,
     "http_method": None, "file": "OwnerRepository.java", "text": "interface for owners"},
]


@pytest.fixture(autouse=True)
def _clean_lexical_index():
    reset_lexical_index()
    yield
    reset_lexical_index()


def result(identifier, result_type="file", **extra):
    return {"id": identifier, "type": result_type, **extra}


def test_type_priority_ranks_endpoints_above_containers():
    assert type_priority(result(1, "endpoint")) > type_priority(result(2, "method"))
    assert type_priority(result(2, "method")) > type_priority(result(3, "class"))
    assert type_priority(result(3, "class")) > type_priority(result(4, "file"))


def test_fuse_promotes_a_result_both_retrievers_rank():
    """The point of fusion: agreement beats a single strong opinion."""
    vector = [result(1), result(2), result(3)]
    lexical = [result(4), result(2), result(5)]

    fused = fuse([vector, lexical], top_k=3)

    assert fused[0]["id"] == 2


def test_fuse_breaks_ties_by_type_priority():
    vector = [result(1, "file"), result(2, "endpoint")]
    lexical = [result(2, "endpoint"), result(1, "file")]

    fused = fuse([vector, lexical], top_k=2)

    assert fused[0]["id"] == 2


def test_fuse_deduplicates_across_rankings():
    fused = fuse([[result(1), result(2)], [result(1), result(2)]], top_k=10)

    assert [item["id"] for item in fused] == [1, 2]


def test_fuse_respects_top_k():
    rankings = [[result(index) for index in range(10)]]

    assert len(fuse(rankings, top_k=3)) == 3


def test_fuse_handles_empty_rankings():
    assert fuse([[], []], top_k=5) == []


def test_lexical_index_is_built_once_and_rebuilt_after_a_reset():
    with patch("app.retrieval.iter_points", return_value=INDEXED) as points:
        get_lexical_index()
        get_lexical_index()
        assert points.call_count == 1, "the index should be cached"

        reset_lexical_index()
        get_lexical_index()
        assert points.call_count == 2, "a reset must force a rebuild"


def test_lexical_search_matches_a_route_held_only_in_metadata():
    """The route is searchable even though it is not in the chunk body."""
    with patch("app.retrieval.iter_points", return_value=INDEXED):
        assert lexical_search("/owners/new", top_k=2)[0]["id"] == 1


def test_lexical_search_matches_words_inside_a_symbol():
    with patch("app.retrieval.iter_points", return_value=INDEXED):
        assert lexical_search("creation form", top_k=2)[0]["id"] == 1


def test_hybrid_search_combines_both_retrievers():
    with (
        patch("app.retrieval.iter_points", return_value=INDEXED),
        patch("app.retrieval.get_embedding", return_value=[0.1, 0.2]),
        patch("app.retrieval.search", return_value=[INDEXED[1]]) as vector,
    ):
        results = hybrid_search("owner repository", top_k=2)

    assert {item["id"] for item in results} == {1, 2}
    assert vector.called, "the dense retriever must still be consulted"
