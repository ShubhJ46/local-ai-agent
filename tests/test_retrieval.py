from app.retrieval import fuse, type_priority


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
