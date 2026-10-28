"""Tokenisation, sentence splitting and chunking helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how i if in into is it its of on or
    that the their then there these this to was were what when where which who why will
    with you your can do does did not no so than too very about over under also each other
    some such only own same s t""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _collapse_double(stem: str) -> str:
    """'runn' -> 'run' (but keep 'ss', 'll', 'zz': 'pass', 'call')."""
    if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
        return stem[:-1]
    return stem


def _stem(token: str) -> str:
    """A tiny suffix stemmer: 'servers' -> 'server', 'sharing'/'share' -> 'shar'."""
    t = token
    if len(t) > 4 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) > 5 and t.endswith("ing"):
        t = _collapse_double(t[:-3])
    elif len(t) > 4 and t.endswith("ed"):
        t = _collapse_double(t[:-2])
    elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    if len(t) > 4 and t.endswith("e"):
        t = t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop stopwords and apply light stemming."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation and on line breaks."""
    return [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p.strip()]


@dataclass(frozen=True)
class Chunk:
    """A piece of a source document."""

    text: str
    source: str
    index: int  

def chunk_text(text: str, source: str, chunk_size: int = 120, overlap: int = 30) -> list[Chunk]:
    """Group whole sentences into chunks of at most ~`chunk_size` words.

    The last sentences of a chunk (up to `overlap` words) are repeated at the start of the
    next one, so an answer that sits on a boundary is still found intact. A single sentence
    longer than `chunk_size` words is cut into pieces of `chunk_size` words.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    pieces: list[str] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        if len(words) <= chunk_size:
            pieces.append(sentence)
        else:
            pieces.extend(" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size))

    chunks: list[Chunk] = []
    current: list[str] = []
    count = 0
    for piece in pieces:
        n = len(piece.split())
        if current and count + n > chunk_size:
            chunks.append(Chunk(" ".join(current), source, len(chunks)))
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                prev_n = len(prev.split())
                if carried + prev_n > overlap:
                    break
                carry.insert(0, prev)
                carried += prev_n
            current, count = carry, carried
        current.append(piece)
        count += n
    if current:
        chunks.append(Chunk(" ".join(current), source, len(chunks)))
    return chunks
