import requests

from app.config import settings
from app.errors import OllamaError


def query_llm(prompt: str, model: str | None = None) -> str:
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": model or settings.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 180},
            },
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        answer = response.json().get("response")
    except (requests.RequestException, ValueError) as error:
        raise OllamaError(
            "Could not get a response from Ollama. Start Ollama with `ollama serve` "
            f"and pull `{model or settings.llm_model}`."
        ) from error

    if not isinstance(answer, str) or not answer.strip():
        raise OllamaError("Ollama returned a response without text.")
    return answer.strip()


if __name__ == "__main__":
    prompt = "Explain vector databases in simple terms"
    print(query_llm(prompt))
