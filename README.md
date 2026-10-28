# RAG Document Q&A

Ask questions about your own text documents. This project builds a **Retrieval-Augmented Generation (RAG)** pipeline from the ground up: it splits documents into chunks, finds the chunks most relevant to your question, and then asks a local **Ollama** language model to answer using only those chunks.

If Ollama is not installed, the tool still works: it falls back to an offline answerer that returns the best-matching sentences.

```
documents ──► chunking ──► BM25 retrieval ──► prompt with context ──► Ollama LLM ──► answer + sources
                                                                          │
                                                          (offline fallback: best matching sentences)
```

## Features

- **Sentence-aware chunking** with overlap, so answers on a chunk boundary are not cut in half
- **BM25 retriever** written from scratch (no vector database or external service needed)
- **Grounded prompts:** the model is told to use only the retrieved context and to say "I don't know" otherwise
- **Ollama client** using only the Python standard library (`urllib`)
- **Automatic fallback** to an offline extractive answerer if Ollama is missing or fails
- **Sources shown** with every answer, plus `--show-context` to see the retrieved chunks
- Interactive chat mode and single-question mode
- Zero runtime dependencies. `pytest` is only needed to run the tests.

## Quick start

```bash
git clone https://github.com/Payal-ok/rag-document-qa.git
cd rag-document-qa

# Optional but recommended: install Ollama from https://ollama.com and pull a model
ollama pull llama3.2

# Ask a question about the sample documents
python main.py "What happens if the learning rate is too large?"

# Interactive mode
python main.py

# Use your own notes (any folder of .txt / .md files)
python main.py --docs ~/my_notes "Summarise the deployment steps"

# Force the offline answerer (no Ollama needed)
python main.py --no-llm "How do I activate a virtual environment on Windows?"
```

Example offline output:

```
$ python main.py --no-llm "How do I activate a virtual environment on Windows?"

On Windows, run `.venv\Scripts\activate`.

[answered by: extractive]
Sources: python_virtual_environments.md
```

### Options

| Flag | Meaning | Default |
|---|---|---|
| `--docs` | Folder with `.txt` / `.md` files | `sample_docs/` |
| `--model` | Ollama model name | `llama3.2` |
| `--host` | Ollama server URL | `http://localhost:11434` |
| `-k` | Number of chunks to retrieve | 3 |
| `--chunk-size`, `--overlap` | Chunking, measured in words | 120, 30 |
| `--no-llm` | Skip Ollama | off |
| `--show-context` | Print the retrieved chunks | off |

## Use it as a library

```python
from rag import RAGPipeline, OllamaClient

pipeline = RAGPipeline("sample_docs", llm=OllamaClient(model="llama3.2"))
answer = pipeline.ask("What does the 404 status code mean?")

print(answer.text)      # the generated answer
print(answer.sources)   # ['rest_apis.md']
print(answer.mode)      # 'ollama' or 'extractive'
```

## How it works

1. **Load:** `rag/loader.py` reads `.txt` and `.md` files. Markdown headings are joined to the paragraph below them so their keywords stay with the text.
2. **Chunk:** `rag/text.py` groups whole sentences into chunks of about 120 words, repeating up to 30 words between neighbouring chunks.
3. **Retrieve:** `rag/retriever.py` scores every chunk against the question with the **BM25** ranking formula (term frequency, inverse document frequency and length normalisation) and returns the top *k*.
4. **Generate:** `rag/llm.py` builds a prompt containing the numbered chunks and sends it to Ollama's `/api/generate` endpoint. If the server is unreachable, the offline answerer picks the sentences that best match the rarest words in the question.
5. **Answer:** `rag/pipeline.py` returns the answer, the mode used, and the source documents.

## Project structure

```
rag-document-qa/
├── main.py                 # command line interface
├── rag/
│   ├── text.py             # tokenizer, sentence splitter, chunker
│   ├── loader.py           # reads documents
│   ├── retriever.py        # BM25
│   ├── llm.py              # Ollama client, offline answerer, prompt builder
│   └── pipeline.py         # ties everything together
├── sample_docs/            # three small example documents
└── tests/test_rag.py       # 25 tests
```

## Tests

```bash
pip install pytest
python -m pytest
```

The tests cover chunking, tokenisation, BM25 ranking, the full pipeline on the sample documents, error handling (missing folder, empty question), and the Ollama client. The Ollama client is tested against a small **fake local HTTP server**, so the tests run without a real model.

## Limitations and ideas for improvement

- BM25 matches words, not meaning. Searching with **embeddings** (for example from an Ollama embedding model) would also find paraphrases.
- Only `.txt` and `.md` files are supported. PDF and DOCX loaders would be a good addition.
- The index is rebuilt on every run, which is fine for small folders. For large collections, save it to disk.
- Re-implement the pipeline with **LangChain** and compare the results with this from-scratch version.

## License

MIT
