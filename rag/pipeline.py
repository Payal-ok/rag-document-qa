"""The end-to-end RAG pipeline: load -> chunk -> retrieve -> generate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .llm import ExtractiveAnswerer, OllamaClient, OllamaError, build_prompt
from .loader import load_documents
from .retriever import BM25Retriever, Hit
from .text import chunk_text


@dataclass
class Answer:
    question: str
    text: str
    mode: str 
    hits: list[Hit] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        seen = []
        for hit in self.hits:
            if hit.chunk.source not in seen:
                seen.append(hit.chunk.source)
        return seen


class RAGPipeline:
    """Answer questions about a folder of text documents.

    Parameters
    ----------
    docs_dir : folder with .txt / .md files
    llm : an OllamaClient, or None to always use the offline extractive answerer
    chunk_size, overlap : chunking settings, measured in words
    """

    def __init__(
        self,
        docs_dir: str | Path,
        llm: OllamaClient | None = None,
        chunk_size: int = 120,
        overlap: int = 30,
    ):
        self.llm = llm
        self.fallback = ExtractiveAnswerer()
        documents = load_documents(docs_dir)
        self.chunks = []
        for name, text in documents.items():
            self.chunks.extend(chunk_text(text, source=name, chunk_size=chunk_size, overlap=overlap))
        self.retriever = BM25Retriever(self.chunks)

    def ask(self, question: str, k: int = 3) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        hits = self.retriever.search(question, k=k)
        if not hits:
            return Answer(question, "I don't know based on the documents.", "extractive", [])

        chunks = [h.chunk for h in hits]
        if self.llm is not None:
            try:
                text = self.llm.generate(build_prompt(question, chunks))
                if text:
                    return Answer(question, text, "ollama", hits)
            except OllamaError:
                pass  # fall through to the offline answer
        return Answer(question, self.fallback.answer(question, chunks), "extractive", hits)
