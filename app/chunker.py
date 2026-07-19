def chunk_text(text: str, chunk_size: int = 1_200, overlap: int = 200) -> list[str]:
    """Split text into deterministic, overlapping character windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be greater than or equal to 0 and smaller than chunk_size")
    if not text:
        return []

    step = chunk_size - overlap
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def chunk_code_unit(text: str, max_chars: int = 4_000) -> list[str]:
    """Keep a code unit whole unless it exceeds the embedding budget.

    A function or endpoint is already a meaningful retrieval unit, so splitting it
    into fixed character windows destroys exactly the structure the AST recovered.
    Oversized units split on line boundaries instead, so no fragment begins
    mid-statement.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    size = 0

    for line in text.splitlines(keepends=True):
        # A single line longer than the budget cannot be split at a statement
        # boundary, so it is emitted whole rather than cut arbitrarily.
        if current and size + len(line) > max_chars:
            parts.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)

    if current:
        parts.append("".join(current))
    return parts
