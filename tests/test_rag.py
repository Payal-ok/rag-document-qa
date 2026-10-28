import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from rag import (
    BM25Retriever,
    ExtractiveAnswerer,
    OllamaClient,
    OllamaError,
    RAGPipeline,
    build_prompt,
    chunk_text,
    tokenize,
)

SAMPLE_DOCS = Path(__file__).parent.parent / "sample_docs"


# ---------- text helpers ----------
def test_tokenize_removes_stopwords_and_stems():
    assert tokenize("The servers are running") == ["server", "run"]
    assert tokenize("The cat") == ["cat"]
    assert tokenize("Packages") == tokenize("package")


def test_chunk_text_overlap_and_coverage():
    sentences = [" ".join(f"w{i * 10 + j}" for j in range(10)) + "." for i in range(10)]
    chunks = chunk_text(" ".join(sentences), "doc.txt", chunk_size=30, overlap=10)
    assert len(chunks) > 1
    
    for c in chunks:
        assert len(c.text.split()) <= 30
        assert c.text.endswith(".")
    
    assert chunks[0].text.split(". ")[-1].rstrip(".") == chunks[1].text.split(". ")[0].rstrip(".")
    
    covered = set(" ".join(c.text for c in chunks).replace(".", "").split())
    assert covered == {f"w{i}" for i in range(100)}
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_text_splits_overlong_sentence():
    text = " ".join(f"w{i}" for i in range(100))  
    chunks = chunk_text(text, "doc.txt", chunk_size=30, overlap=10)
    assert [len(c.text.split()) for c in chunks] == [30, 30, 30, 10]


def test_chunk_text_short_and_empty():
    assert len(chunk_text("just a few words", "d", chunk_size=50, overlap=5)) == 1
    assert chunk_text("   ", "d") == []


def test_chunk_text_validation():
    with pytest.raises(ValueError):
        chunk_text("a b c", "d", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("a b c", "d", chunk_size=5, overlap=5)


# ---------- retrieval ----------
def test_bm25_ranks_relevant_chunk_first():
    docs = {
        "cooking.txt": "Boil the pasta in salted water and add tomato sauce.",
        "space.txt": "The telescope observed a distant galaxy and a bright star.",
        "code.txt": "Python functions return values and use indentation for blocks.",
    }
    chunks = [c for name, t in docs.items() for c in chunk_text(t, name)]
    hits = BM25Retriever(chunks).search("which telescope saw the galaxy", k=2)
    assert hits[0].chunk.source == "space.txt"


def test_bm25_returns_nothing_for_unrelated_query():
    chunks = chunk_text("apples and oranges grow on trees", "fruit.txt")
    assert BM25Retriever(chunks).search("quantum chromodynamics") == []


def test_bm25_needs_chunks():
    with pytest.raises(ValueError):
        BM25Retriever([])


@pytest.fixture(scope="module")
def pipeline():
    return RAGPipeline(SAMPLE_DOCS, llm=None)


@pytest.mark.parametrize(
    "question, expected_source, expected_word",
    [
        ("How do I activate a virtual environment on Windows?", "python_virtual_environments.md", "activate"),
        ("What does the 404 status code mean?", "rest_apis.md", "404"),
        ("What happens if the learning rate is too large?", "machine_learning_basics.md", "diverge"),
        ("Which HTTP method creates a new resource?", "rest_apis.md", "POST"),
        ("How can I detect overfitting?", "machine_learning_basics.md", "test set"),
    ],
)
def test_pipeline_finds_right_document(pipeline, question, expected_source, expected_word):
    answer = pipeline.ask(question)
    assert answer.sources[0] == expected_source
    assert expected_word.lower() in answer.text.lower()
    assert answer.mode == "extractive"


def test_pipeline_admits_when_it_does_not_know(pipeline):
    answer = pipeline.ask("Who won the football world cup in 1998?")
    assert "don't know" in answer.text.lower()
    assert answer.hits == []


def test_pipeline_rejects_empty_question(pipeline):
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_pipeline_missing_folder():
    with pytest.raises(FileNotFoundError):
        RAGPipeline("does/not/exist")


def test_pipeline_empty_folder(tmp_path):
    with pytest.raises(ValueError):
        RAGPipeline(tmp_path)



def test_prompt_contains_context_and_question():
    chunks = chunk_text("Cats sleep a lot.", "pets.txt")
    prompt = build_prompt("Do cats sleep?", chunks)
    assert "Cats sleep a lot." in prompt
    assert "Do cats sleep?" in prompt
    assert "pets.txt" in prompt
    assert "ONLY" in prompt


def test_extractive_answerer_picks_matching_sentence():
    chunks = chunk_text("The sky is blue. Grass is green. Roses are red.", "colors.txt")
    answer = ExtractiveAnswerer(max_sentences=1).answer("What colour is grass", chunks)
    assert answer == "Grass is green."



class _FakeOllama(BaseHTTPRequestHandler):
    last_request = None

    def log_message(self, *args):  
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": "llama3.2:latest"}]})
        else:
            self._send(404, {})

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        _FakeOllama.last_request = body
        if body["model"] == "broken":
            self._send(200, {"error": "model failed to load"})
        else:
            self._send(200, {"response": "  Fake answer from the model [1].  "})


@pytest.fixture()
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_ollama_client_generate_and_availability(fake_server):
    client = OllamaClient(model="llama3.2", host=fake_server)
    assert client.is_available()
    assert client.generate("hello") == "Fake answer from the model [1]."
    assert _FakeOllama.last_request["stream"] is False
    assert _FakeOllama.last_request["prompt"] == "hello"


def test_ollama_client_missing_model(fake_server):
    assert not OllamaClient(model="mistral", host=fake_server).is_available()


def test_ollama_client_reports_server_errors(fake_server):
    with pytest.raises(OllamaError):
        OllamaClient(model="broken", host=fake_server).generate("hi")


def test_ollama_client_unreachable_server():
    client = OllamaClient(host="http://127.0.0.1:9", timeout=1)
    assert not client.is_available()
    with pytest.raises(OllamaError):
        client.generate("hi")


def test_pipeline_uses_ollama_when_available(fake_server):
    pipe = RAGPipeline(SAMPLE_DOCS, llm=OllamaClient("llama3.2", fake_server))
    answer = pipe.ask("What does the 404 status code mean?")
    assert answer.mode == "ollama"
    assert answer.text == "Fake answer from the model [1]."
    
    assert "404" in _FakeOllama.last_request["prompt"]


def test_pipeline_falls_back_if_ollama_fails():
    dead = OllamaClient(host="http://127.0.0.1:9", timeout=1)
    answer = RAGPipeline(SAMPLE_DOCS, llm=dead).ask("What does the 404 status code mean?")
    assert answer.mode == "extractive"
    assert "404" in answer.text
