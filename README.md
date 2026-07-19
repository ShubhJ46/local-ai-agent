# Local AI Code Agent

A local-first codebase assistant that indexes source code with AST-aware metadata and answers questions with retrieved source context. It runs with Ollama and an embedded Qdrant store—no cloud API key required.

## What it does today

- Extracts Python functions and Spring Java endpoint metadata with Tree-sitter.
- Indexes supported source and text files into embedded Qdrant.
- Retrieves relevant code with semantic search and lightweight metadata-aware reranking.
- Sends retrieved source, paths, symbols, and endpoint details to a local LLM for grounded answers.
- Provides a terminal workflow for indexing a repository and asking questions.
- Includes offline tests, linting, CI, and a versioned retrieval evaluation harness.

## Architecture

```text
CLI question
  -> query rewrite
  -> Ollama embedding
  -> Qdrant semantic retrieval
  -> metadata reranking
  -> source-grounded Ollama response
```

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
python3 -m scripts.evaluate_retrieval data
```

The evaluation command reports Recall@k and MRR against `evaluation/retrieval_cases.json`. Add representative questions from a real target codebase before publishing any metrics.

## Roadmap

- Hybrid lexical + vector retrieval with reciprocal-rank fusion.
- Call graph extraction (controller -> service -> repository).
- Precise citations with file and line ranges.
- More language extractors and a browser UI.

## Project story

This project demonstrates practical retrieval-system engineering: parsing and metadata extraction, local model integration, vector indexing, deterministic evaluation, error handling, and a tested command-line product. It is intentionally local-first so the full workflow can be run and inspected without sending a codebase to a third-party API.
