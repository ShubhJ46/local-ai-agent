import requests

def get_embedding(text):
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            },
            timeout=30
        )

        response.raise_for_status()
        return response.json()["embedding"]

    except requests.exceptions.ConnectionError:
        raise Exception("Ollama is not running. Start it with: ollama serve")
    

if __name__ == "__main__":
    e1 = get_embedding("login system")
    e2 = get_embedding("authentication module")
    print(len(e1), len(e2))