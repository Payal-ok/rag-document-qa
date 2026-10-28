"""Command line interface for the document Q&A tool.

Examples
--------
python main.py "What is a virtual environment?"
python main.py --docs my_notes --model llama3.2 --show-context
python main.py                      # interactive mode
python main.py --no-llm "What is REST?"   # force the offline answerer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag import OllamaClient, RAGPipeline

DEFAULT_DOCS = Path(__file__).parent / "sample_docs"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ask questions about your own text documents.")
    p.add_argument("question", nargs="?", help="Question to ask. Omit for interactive mode.")
    p.add_argument("--docs", default=str(DEFAULT_DOCS), help="Folder with .txt / .md files")
    p.add_argument("--model", default="llama3.2", help="Ollama model name")
    p.add_argument("--host", default="http://localhost:11434", help="Ollama server URL")
    p.add_argument("-k", type=int, default=3, help="How many chunks to retrieve")
    p.add_argument("--chunk-size", type=int, default=120, help="Chunk size in words")
    p.add_argument("--overlap", type=int, default=30, help="Chunk overlap in words")
    p.add_argument("--no-llm", action="store_true", help="Skip Ollama and use the offline answerer")
    p.add_argument("--show-context", action="store_true", help="Print the retrieved chunks")
    return p.parse_args(argv)


def print_answer(answer, show_context: bool):
    print(f"\n{answer.text}")
    print(f"\n[answered by: {answer.mode}]")
    if answer.sources:
        print("Sources:", ", ".join(answer.sources))
    if show_context:
        for i, hit in enumerate(answer.hits, 1):
            print(f"\n--- chunk {i} | {hit.chunk.source} #{hit.chunk.index} | score {hit.score:.2f}")
            print(hit.chunk.text)


def main(argv=None) -> int:
    args = parse_args(argv)

    llm = None
    if not args.no_llm:
        client = OllamaClient(model=args.model, host=args.host)
        if client.is_available():
            llm = client
        else:
            print(
                f"(Ollama model '{args.model}' not found at {args.host}; "
                "using the offline answerer. Run `ollama pull "
                f"{args.model}` for full answers.)",
                file=sys.stderr,
            )

    try:
        pipeline = RAGPipeline(args.docs, llm=llm, chunk_size=args.chunk_size, overlap=args.overlap)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.question:
        print_answer(pipeline.ask(args.question, k=args.k), args.show_context)
        return 0

    print(f"Loaded {len(pipeline.chunks)} chunks. Type a question, or 'quit' to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            print_answer(pipeline.ask(question, k=args.k), args.show_context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
