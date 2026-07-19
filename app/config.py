"""Runtime configuration for the local code agent."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    # A 3B code model rather than a general 7B. On a machine with a ~5 GiB GPU
    # budget a 7B does not fit alongside its KV cache, and the resulting spill to
    # CPU costs far more than the extra parameters are worth: measured 0.9 tok/s
    # for mistral against 12.7 tok/s here, on the same hardware.
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5-coder:3b")
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./qdrant_data")
    # Embedded Qdrant locks its directory, so only one process may use an index
    # at a time. Point this at a Qdrant server to share one index between the
    # CLI and the HTTP API.
    qdrant_url: str = os.getenv("QDRANT_URL", "")


settings = Settings()
