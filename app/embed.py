from functools import lru_cache

import requests

from app.config import settings
from app.errors import OllamaError

# The same text is often embedded more than once in a single question: to recall
# memory, to search, and again inside a tool. Embeddings are deterministic for a
# fixed model, and a repeat call is not merely slow -- when the chat and
# embedding models do not both fit in VRAM, each one evicts the other.
EMBEDDING_CACHE_SIZE = 512


@lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
def _embed_cached(text: str) -> tuple[float, ...]:
    return tuple(_embed(text))


def get_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    return list(_embed_cached(text))


def _embed(text: str) -> list[float]:
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/embeddings",
            json={"model": settings.embedding_model, "prompt": text},
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")
    except (requests.RequestException, ValueError) as error:
        raise OllamaError(
            "Could not get an embedding from Ollama. Start Ollama with `ollama serve` "
            f"and pull `{settings.embedding_model}`."
        ) from error

    if not isinstance(embedding, list) or not embedding:
        raise OllamaError("Ollama returned a response without a non-empty embedding.")
    return embedding


if __name__ == "__main__":
    e1 = get_embedding("login system")
    e2 = get_embedding("authentication module")
    print(len(e1), len(e2))
