import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import app, get_llm, get_pipeline, get_store
from app.llm import NOT_FOUND, build_user_message
from app.rag import RagPipeline
from app.vector_store import VectorStore


class FakeLLM:
    """Records what it was asked and echoes the first retrieved chunk - no API calls in tests."""

    def __init__(self):
        self.calls = []

    def answer(self, question, hits):
        self.calls.append((question, hits))
        return f"From [1]: {hits[0].text[:40]}"


@pytest.fixture
def store(embedding):
    # The in-memory Chroma client is shared per process, so give each test its own collection.
    return VectorStore(path=None, collection=f"test_{uuid.uuid4().hex}", embedding_function=embedding)


@pytest.fixture
def client(store):
    llm = FakeLLM()
    app.dependency_overrides[get_store] = lambda: store
    # The hash test embedding gives larger distances than a real model, so relax the relevance cut-off.
    app.dependency_overrides[get_pipeline] = lambda: RagPipeline(store, llm, max_distance=2.0)
    yield TestClient(app), llm
    app.dependency_overrides.clear()


def test_ingest_then_ask_returns_answer_with_page_citations(client, manual_bytes):
    http, llm = client
    r = http.post("/ingest", files={"file": ("manual.pdf", manual_bytes, "application/pdf")})
    assert r.status_code == 200
    assert r.json()["pages"] == 6

    r = http.post("/ask", json={"question": "What does error code E03 mean?"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"], "expected retrieved sources"
    assert 5 in [s["page"] for s in body["sources"]]  # ranking quality is measured in eval/, not here
    assert len(llm.calls) == 1


def test_reingesting_a_document_replaces_its_chunks(client, store, manual_bytes):
    http, _ = client
    first = http.post("/ingest", files={"file": ("manual.pdf", manual_bytes)}).json()["chunks"]
    http.post("/ingest", files={"file": ("manual.pdf", manual_bytes)})
    assert store.count() == first


def test_ingest_rejects_non_pdf(client):
    http, _ = client
    r = http.post("/ingest", files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 400


def test_ask_validates_question_length(client):
    http, _ = client
    assert http.post("/ask", json={"question": "x"}).status_code == 422


def test_pipeline_returns_not_found_without_calling_llm_when_nothing_relevant(store):
    llm = FakeLLM()
    result = RagPipeline(store, llm).ask("What is the price?")  # empty store
    assert result.answer == NOT_FOUND
    assert result.sources == []
    assert llm.calls == []


def test_prompt_numbers_passages_and_includes_page_metadata(store, manual_bytes):
    from app.ingest import chunk_document

    store.add_chunks(chunk_document(manual_bytes, "manual.pdf"))
    hits = store.search("calibration interval", k=2)
    message = build_user_message("How often to calibrate?", hits)
    assert "[1] (source: manual.pdf, page" in message
    assert "[2]" in message
    assert message.endswith("Question: How often to calibrate?")
