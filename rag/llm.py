"""Language model backends: a local Ollama server, and an offline extractive fallback."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections import Counter

from .text import Chunk, split_sentences, tokenize

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Raised when the Ollama server cannot be reached or returns an error."""


class OllamaClient:
    """Minimal client for the Ollama HTTP API (https://github.com/ollama/ollama/blob/main/docs/api.md).

    Run a model locally first, for example:  ollama pull llama3.2
    """

    def __init__(self, model: str = "llama3.2", host: str = DEFAULT_HOST, timeout: float = 120.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """True if the server answers and the chosen model is installed."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as resp:
                data = json.load(resp)
        except (urllib.error.URLError, OSError, ValueError):
            return False
        names = [m.get("name", "") for m in data.get("models", [])]
        return any(n == self.model or n.split(":")[0] == self.model.split(":")[0] for n in names)

    def generate(self, prompt: str) -> str:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.load(resp)
        except urllib.error.HTTPError as exc:
            raise OllamaError(f"Ollama returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaError(f"Could not reach Ollama at {self.host}: {exc}") from exc
        except ValueError as exc:
            raise OllamaError("Ollama sent a response that is not valid JSON") from exc

        if "error" in body:
            raise OllamaError(str(body["error"]))
        return str(body.get("response", "")).strip()


class ExtractiveAnswerer:
    """Offline fallback: return the sentences that best overlap with the question.

    This needs no model, so the project works even without Ollama installed.
    """

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        q_terms = set(tokenize(question))

       
        sentences, seen = [], set()
        for chunk in chunks:
            for sentence in split_sentences(chunk.text):
                if sentence not in seen:
                    seen.add(sentence)
                    sentences.append((sentence, q_terms & set(tokenize(sentence))))

        
        n = len(sentences)
        doc_freq = {t: sum(1 for _, terms in sentences if t in terms) for t in q_terms}
        weight = {t: math.log(1 + n / df) for t, df in doc_freq.items() if df}

        scored = [(sum(weight[t] for t in terms), sentence) for sentence, terms in sentences if terms]
        if not scored:
            return "I could not find an answer in the documents."

        
        ranked = sorted(scored, key=lambda pair: (-pair[0], len(pair[1])))
        top_score = ranked[0][0]
        keep = {s for sc, s in ranked[: self.max_sentences] if sc >= 0.7 * top_score}
        return " ".join(s for _, s in scored if s in keep)


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Create a grounded prompt that tells the model to use only the given context."""
    context = "\n\n".join(f"[{i}] (source: {c.source})\n{c.text}" for i, c in enumerate(chunks, 1))
    return (
        "You are a helpful assistant that answers questions using ONLY the context below.\n"
        "If the context does not contain the answer, say \"I don't know based on the documents.\"\n"
        "Keep the answer short and mention the source numbers like [1] that you used.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
