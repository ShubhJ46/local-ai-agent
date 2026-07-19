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
