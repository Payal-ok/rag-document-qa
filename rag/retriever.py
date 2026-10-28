"""BM25 retrieval implemented from scratch."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .text import Chunk, tokenize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Retriever:
    """Rank chunks against a query with the Okapi BM25 formula.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("Cannot build a retriever with no chunks")
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self._term_freqs = [Counter(tokenize(c.text)) for c in chunks]
        self._doc_lens = [sum(tf.values()) for tf in self._term_freqs]
        self._avgdl = sum(self._doc_lens) / len(chunks) or 1.0

        doc_freq: Counter = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def _score(self, query_terms: list[str], doc_id: int) -> float:
        tf = self._term_freqs[doc_id]
        length_norm = 1 - self.b + self.b * self._doc_lens[doc_id] / self._avgdl
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f:
                score += self._idf[term] * f * (self.k1 + 1) / (f + self.k1 * length_norm)
        return score

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Return up to k chunks with a positive score, best first."""
        terms = tokenize(query)
        if not terms:
            return []
        scored = [(self._score(terms, i), i) for i in range(len(self.chunks))]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [Hit(self.chunks[i], s) for s, i in scored[:k] if s > 0]
