import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def query_llm(prompt, model="mistral"):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


if __name__ == "__main__":
    prompt = "Explain vector databases in simple terms"
    print(query_llm(prompt))