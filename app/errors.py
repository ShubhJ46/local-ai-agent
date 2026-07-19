class LocalAgentError(RuntimeError):
    """Base exception for errors users can recover from."""


class OllamaError(LocalAgentError):
    """Raised when Ollama cannot serve a valid response."""
