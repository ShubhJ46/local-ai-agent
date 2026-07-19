# Local AI Code Agent

A local-first codebase assistant that indexes source code with AST-aware metadata and answers questions with retrieved source context. It runs with Ollama and an embedded Qdrant store—no cloud API key required.

## What it does today

- Extracts Python functions and Spring Java types, methods, and endpoint routes with Tree-sitter.
- Embeds each AST unit whole, with its symbol, route, and line range as metadata.
- Retrieves with hybrid search: dense vectors fused with BM25, then prioritised by symbol type.
- Answers through a bounded tool-use loop that is grounded on retrieved source before it may reply.
- Remembers both the current conversation and, in Qdrant, exchanges from earlier sessions.
- Provides a terminal workflow for indexing a repository and asking questions.
- Includes offline tests, linting, CI, and a versioned retrieval evaluation harness.

## Architecture

```text
CLI question
  -> Ollama embedding ---> Qdrant vector search --.
  |                                                >-- reciprocal rank fusion
  '-> tokenisation ------> BM25 lexical search ---'         |
                                                            v
                                                    type prioritisation
                                                    (endpoint > method
                                                     > class > file)
                                                            |
                                                            v
                                              source-grounded agent loop
                                              (tools, bounded steps,
                                               short- + long-term memory)
                                                            |
                                                            v
                                                    cited answer
```

## Retrieval quality

Measured against [spring-petclinic](https://github.com/spring-projects/spring-petclinic)
(pinned revision, 115 indexed chunks) using 30 questions in
`evaluation/retrieval_cases.json`, at k=5:

| Configuration | Recall@5 | MRR |
| --- | --- | --- |
| Vector only | 76.7% | 0.594 |
| BM25 only | 50.0% | 0.354 |
| **Hybrid (shipped)** | **86.7%** | **0.662** |

Endpoint-navigation questions reach 100% Recall@5. The starting point before
this work — character-window chunking with keyword reranking — scored 70.0% /
0.503, so hybrid retrieval over whole AST units is worth **+16.7 points of
recall and +32% MRR**.

Reproduce it yourself:

```bash
./scripts/fetch_eval_corpus.sh
python3 -m scripts.evaluate_retrieval .eval-corpus/spring-petclinic/src/main
```

A sweep over fusion parameters (candidate pool size, RRF damping) found no
setting that reliably beat the defaults, so the shipped constants are the
standard ones rather than values fitted to this question set.

## Quick start

Prerequisites: Python 3.10+ and [Ollama](https://ollama.com/).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
ollama serve
ollama pull nomic-embed-text
ollama pull mistral
python3 -m app.cli
```

In the CLI:

```text
load /absolute/path/to/a/codebase
Where is authentication handled?
exit
```

Copy `.env.example` to `.env` to choose an Ollama endpoint, models, timeout, or local Qdrant directory.

## Supported inputs

Python functions and Spring Java controllers receive structured metadata. C/C++, JavaScript, TypeScript, Markdown, and text files are currently indexed as file-level documents. The next technical milestone is adding AST extractors and symbol/chunk metadata for the remaining languages.

## Quality checks

```bash
ruff check .
pytest
./scripts/fetch_eval_corpus.sh
python3 -m scripts.evaluate_retrieval .eval-corpus/spring-petclinic/src/main
```

Tests are fully offline — Ollama calls are mocked — so `ruff` and `pytest` gate
CI. The evaluation command needs a running Ollama and reports Recall@k and MRR
per retriever and per question category.

## Known limitations

- Only Python and Java have AST extractors. C/C++, JavaScript, TypeScript, Markdown, and text files are indexed as whole-file documents.
- Embedding is one request per chunk, so first-time indexing of a large repository is slow.
- Re-indexing replaces the collection, so only one codebase is searchable at a time.
- The BM25 index is held in memory and rebuilt from the vector store on first query.
- Answer quality is bounded by the local model, and this is the weakest link. Retrieval reaches 86.7% Recall@5, but `mistral` at temperature 0 still sometimes cites a plausible-sounding path that is not in the retrieved source. Grounding the loop on retrieved source before it may answer removed this on direct questions; it still occurs on follow-ups. Verifying cited file names against the index is the next thing worth building.

### A note on hardware

If the chat and embedding models do not both fit in VRAM, Ollama evicts one to
load the other. On an 8 GB M1 with `mistral` + `nomic-embed-text`, a warm
embedding takes 0.1s but the first one *after* a generation takes ~20s, because
the model is reloaded. Since the agent alternates between the two, this
dominates latency.

Two things in this repo reduce it: embeddings are memoised per process, and
long-term recall is computed once per question rather than once per agent step.
To avoid the swap entirely, run a smaller chat model, or keep both resident:

```bash
OLLAMA_MAX_LOADED_MODELS=2 ollama serve
```

## Roadmap

- Batched, incremental indexing keyed on content hashes.
- Call graph extraction (controller -> service -> repository).
- Line-range citations surfaced in the answer text, not just in retrieval metadata.
- More language extractors and a browser UI.

## Project story

This project demonstrates practical retrieval-system engineering: parsing and metadata extraction, local model integration, vector indexing, deterministic evaluation, error handling, and a tested command-line product. It is intentionally local-first so the full workflow can be run and inspected without sending a codebase to a third-party API.
