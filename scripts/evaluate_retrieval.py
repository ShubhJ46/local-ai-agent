"""Measure code-retrieval quality against a versioned question set.

Reports Recall@k and MRR for each retriever so the contribution of every
component is visible rather than assumed:

  vector  dense embedding search alone
  bm25    lexical search alone
  hybrid  the two rankings fused, then type-prioritised

Run against a real codebase, not the toy fixtures in data/:

    scripts/fetch_eval_corpus.sh
    python3 -m scripts.evaluate_retrieval .eval-corpus/spring-petclinic/src/main
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.ingest.ingest import ingest_codebase
from app.retrieval import hybrid_search, lexical_search, vector_search
from app.vector_store import close_client

CONFIGURATIONS = {
    "vector": vector_search,
    "bm25": lexical_search,
    "hybrid": hybrid_search,
}

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
    outcomes: dict[str, list[tuple[dict, int | None]]] = {name: [] for name in CONFIGURATIONS}

    for case in cases:
        for name, retrieve in CONFIGURATIONS.items():
            results = retrieve(case["question"], top_k)
            outcomes[name].append((case, rank_of_first_hit(results, case)))

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

    names = list(CONFIGURATIONS)
    header = "".join(f"{name:>10}" for name in names)
    print(f"{'case':<28}{header}")
    print("-" * (28 + 10 * len(names)))
    for index, (case, _rank) in enumerate(outcomes[names[0]]):
        ranks = "".join(f"{outcomes[name][index][1] or '-':>10}" for name in names)
        print(f"{case['id']:<28}{ranks}")

    print(f"\n{'configuration':<12} {'Recall@k':>10} {'MRR':>8}")
    print("-" * 32)
    for name in names:
        recall, mrr = summarize(outcomes[name])
        print(f"{name:<12} {recall:>9.1%} {mrr:>8.3f}")

    print(f"\n{'category':<12} {'n':>3}{header}   (Recall@k)")
    print("-" * (15 + 10 * len(names)))
    by_category: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for name in names:
        for case, rank in outcomes[name]:
            by_category[case.get("category", "uncategorized")][name].append((case, rank))
    for category in sorted(by_category):
        counts = by_category[category]
        recalls = "".join(f"{summarize(counts[name])[0]:>9.1%} " for name in names)
        print(f"{category:<12} {len(counts[names[0]]):>3}{recalls}")

    misses = [case["id"] for case, rank in outcomes["hybrid"] if rank is None]
    if misses:
        print(f"\nHybrid misses ({len(misses)}): {', '.join(misses)}")

    close_client()


if __name__ == "__main__":
    main()
