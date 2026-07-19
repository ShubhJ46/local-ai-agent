import requests

from app.config import settings
from app.errors import OllamaError


def get_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

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
