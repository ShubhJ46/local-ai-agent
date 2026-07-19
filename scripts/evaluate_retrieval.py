"""Evaluate retrieval quality against a small, versioned question set."""

import argparse
import json
from pathlib import Path

from app.embed import get_embedding
from app.ingest.ingest import ingest_codebase
from app.tools import rerank, rewrite_query
from app.vector_store import close_client, search


def matches(result: dict, expected: dict) -> bool:
    return (
        expected.get("expected_name") == result.get("name")
        if "expected_name" in expected
        else expected.get("expected_file") == result.get("file")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Recall@k and MRR for code retrieval.")
    parser.add_argument("source", help="Codebase directory to index")
    parser.add_argument("--cases", default="evaluation/retrieval_cases.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    ingest_codebase(args.source)
    reciprocal_ranks = []

    for case in cases:
        query = rewrite_query(case["question"])
        results = rerank(search(get_embedding(query), top_k=args.top_k), case["question"])
        rank = next(
            (index for index, result in enumerate(results, start=1) if matches(result, case)), None
        )
        reciprocal_ranks.append(1 / rank if rank else 0)
        print(f"{'PASS' if rank else 'MISS'} | rank={rank or '-'} | {case['question']}")

    hits = sum(score > 0 for score in reciprocal_ranks)
    total = len(cases)
    print(f"Recall@{args.top_k}: {hits / total:.1%}")
    print(f"MRR@{args.top_k}: {sum(reciprocal_ranks) / total:.3f}")
    close_client()


if __name__ == "__main__":
    main()
