"""rag: a small retrieval-augmented generation pipeline."""

from .llm import ExtractiveAnswerer, OllamaClient, OllamaError, build_prompt
from .pipeline import Answer, RAGPipeline
from .retriever import BM25Retriever
from .text import Chunk, chunk_text, tokenize

__all__ = [
    "RAGPipeline",
    "Answer",
    "BM25Retriever",
    "OllamaClient",
    "OllamaError",
    "ExtractiveAnswerer",
    "build_prompt",
    "Chunk",
    "chunk_text",
    "tokenize",
]
__version__ = "1.0.0"
