"""BM25 lexical search over the indexed chunks.

Embeddings match paraphrase well but drift on exact identifiers: a query naming
``OwnerRepository`` or ``findByLastName`` should rank the chunk that literally
contains that symbol. BM25 supplies the term-matching half of the hybrid
retriever; see :mod:`app.retrieval` for how the two rankings are fused.

Implemented directly rather than pulled from a library so the ranking function
is inspectable and the project keeps no orchestration dependencies.
"""

import math
import re
from collections import Counter

K1 = 1.5
B = 0.75

# Split on non-alphanumerics, then again at camelCase and snake_case boundaries,
# so "processCreationForm" is findable by the words a developer would type.
_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Split source text into lowercased lexical terms.

    Compound identifiers yield both the whole token and its parts, so a query
    can match either the exact symbol or the words inside it.
    """
    tokens = []
    for word in _WORD.findall(text or ""):
        lowered = word.lower()
        tokens.append(lowered)
        pieces = _CAMEL.findall(word)
        if len(pieces) > 1:
            tokens.extend(piece.lower() for piece in pieces)
    return tokens


class BM25Index:
    """An in-memory BM25 index over (identifier, text) pairs."""

    def __init__(self, documents: list[tuple[object, str]]):
        self.ids = [identifier for identifier, _text in documents]
        self.term_frequencies = [Counter(tokenize(text)) for _identifier, text in documents]
        self.lengths = [sum(counts.values()) for counts in self.term_frequencies]
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        document_frequencies: Counter = Counter()
        for counts in self.term_frequencies:
            document_frequencies.update(counts.keys())

        total = len(documents)
        self.inverse_document_frequencies = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, top_k: int = 5) -> list[tuple[object, float]]:
        """Return the top_k (id, score) pairs, best first, omitting zero scores."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_terms = tokenize(query)
        if not query_terms or not self.ids:
            return []

        scored = []
        for index, counts in enumerate(self.term_frequencies):
            score = 0.0
            length_ratio = (
                self.lengths[index] / self.average_length if self.average_length else 0.0
            )
            for term in query_terms:
                frequency = counts.get(term)
                if not frequency:
                    continue
                idf = self.inverse_document_frequencies.get(term, 0.0)
                denominator = frequency + K1 * (1 - B + B * length_ratio)
                score += idf * (frequency * (K1 + 1)) / denominator
            if score > 0:
                scored.append((self.ids[index], score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
