"""Measure code-retrieval quality against a versioned question set.

Reports Recall@k and MRR for two configurations so the contribution of the
reranking layer is visible rather than assumed:

  vector    plain semantic search over the raw question
  pipeline  query rewriting -> semantic search -> reranking

Run against a real codebase, not the toy fixtures in data/:

    scripts/fetch_eval_corpus.sh
    python3 -m scripts.evaluate_retrieval .eval-corpus/spring-petclinic/src/main
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.embed import get_embedding
from app.ingest.ingest import ingest_codebase
from app.tools import rerank, rewrite_query
from app.vector_store import close_client, search

EXPECTED_FIELDS = {"expected_file": "file", "expected_name": "name", "expected_type": "type"}


def matches(result: dict, case: dict) -> bool:
    """A hit must satisfy every expectation the case states, not just one.

    Method names repeat across controllers in real codebases (petclinic has two
    distinct processCreationForm handlers), so name alone is not identifying.
    """
    constraints = [
        (result_key, case[case_key])
        for case_key, result_key in EXPECTED_FIELDS.items()
        if case_key in case
    ]
    if not constraints:
        raise ValueError(f"Case {case.get('id')} states no expectation")
    return all(result.get(result_key) == expected for result_key, expected in constraints)


def rank_of_first_hit(results: list[dict], case: dict) -> int | None:
    return next(
        (index for index, result in enumerate(results, start=1) if matches(result, case)), None
    )


def evaluate(cases: list[dict], top_k: int) -> dict[str, list[tuple[dict, int | None]]]:
    """Return per-configuration (case, rank) pairs."""
    outcomes: dict[str, list[tuple[dict, int | None]]] = {"vector": [], "pipeline": []}

    for case in cases:
        question = case["question"]

        vector_results = search(get_embedding(question), top_k=top_k)
        outcomes["vector"].append((case, rank_of_first_hit(vector_results, case)))

        rewritten = rewrite_query(question)
        pipeline_results = rerank(search(get_embedding(rewritten), top_k=top_k), question)
        outcomes["pipeline"].append((case, rank_of_first_hit(pipeline_results, case)))

    return outcomes


def summarize(ranked: list[tuple[dict, int | None]]) -> tuple[float, float]:
    if not ranked:
        return 0.0, 0.0
    reciprocal = [1 / rank if rank else 0.0 for _case, rank in ranked]
    recall = sum(score > 0 for score in reciprocal) / len(reciprocal)
    return recall, sum(reciprocal) / len(reciprocal)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Recall@k and MRR for code retrieval.")
    parser.add_argument("source", help="Codebase directory to index")
    parser.add_argument("--cases", default="evaluation/retrieval_cases.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    print(f"Indexing {args.source} ...")
    chunk_count = ingest_codebase(args.source)
    print(f"Indexed {chunk_count} chunks. Evaluating {len(cases)} cases at k={args.top_k}.\n")

    outcomes = evaluate(cases, args.top_k)

    print(f"{'case':<28} {'vector':>8} {'pipeline':>10}")
    print("-" * 48)
    for (case, vector_rank), (_case, pipeline_rank) in zip(
        outcomes["vector"], outcomes["pipeline"], strict=True
    ):
        print(f"{case['id']:<28} {vector_rank or '-':>8} {pipeline_rank or '-':>10}")

    print(f"\n{'configuration':<12} {'Recall@k':>10} {'MRR':>8}")
    print("-" * 32)
    for name, ranked in outcomes.items():
        recall, mrr = summarize(ranked)
        print(f"{name:<12} {recall:>9.1%} {mrr:>8.3f}")

    print(f"\n{'category':<12} {'n':>3} {'vector':>10} {'pipeline':>10}   (Recall@k)")
    print("-" * 50)
    by_category: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for name, ranked in outcomes.items():
        for case, rank in ranked:
            by_category[case.get("category", "uncategorized")][name].append((case, rank))
    for category in sorted(by_category):
        counts = by_category[category]
        vector_recall, _ = summarize(counts["vector"])
        pipeline_recall, _ = summarize(counts["pipeline"])
        print(
            f"{category:<12} {len(counts['vector']):>3} "
            f"{vector_recall:>9.1%} {pipeline_recall:>10.1%}"
        )

    misses = [case["id"] for case, rank in outcomes["pipeline"] if rank is None]
    if misses:
        print(f"\nPipeline misses ({len(misses)}): {', '.join(misses)}")

    close_client()


if __name__ == "__main__":
    main()
