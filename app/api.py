"""FastAPI service exposing /ingest and /ask."""
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.ingest import chunk_document
from app.llm import ClaudeLLM
from app.rag import RagPipeline
from app.vector_store import VectorStore

app = FastAPI(title="Engineering Document Q&A Assistant", version="1.0.0")


@lru_cache
def get_store() -> VectorStore:
    return VectorStore(path=settings.chroma_dir)


@lru_cache
def get_llm() -> ClaudeLLM:
    return ClaudeLLM()


def get_pipeline(store: VectorStore = Depends(get_store), llm=Depends(get_llm)) -> RagPipeline:
    return RagPipeline(store, llm)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    source: str | None = Field(default=None, description="Restrict the search to one document")


class Source(BaseModel):
    source: str
    page: int
    section: str
    distance: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health(store: VectorStore = Depends(get_store)):
    return {"status": "ok", "chunks": store.count()}


@app.get("/documents")
def documents(store: VectorStore = Depends(get_store)):
    return {"documents": store.sources()}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), store: VectorStore = Depends(get_store)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    data = await file.read()
    try:
        chunks = chunk_document(data, file.filename, settings.chunk_size, settings.chunk_overlap)
    except Exception as exc:  # corrupt or encrypted PDF
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {exc}") from exc
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text (scanned PDF needs OCR)")
    store.delete_source(file.filename)
    store.add_chunks(chunks)
    return {"document": file.filename, "pages": max(c.page for c in chunks), "chunks": len(chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, pipeline: RagPipeline = Depends(get_pipeline)):
    result = pipeline.ask(request.question, request.source)
    return AskResponse(answer=result.answer, sources=result.sources)
