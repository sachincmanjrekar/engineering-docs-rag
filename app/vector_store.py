"""ChromaDB wrapper: store chunk embeddings and run top-k similarity search."""
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import settings
from app.ingest import Chunk


@dataclass
class SearchHit:
    text: str
    source: str
    page: int
    section: str
    distance: float  # cosine distance: 0 = identical, larger = less similar


class VectorStore:
    def __init__(self, path: str | None = None, collection: str | None = None, embedding_function=None):
        # path=None keeps everything in memory (used by tests and the evaluation script).
        self.client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self.embedding_function = embedding_function or SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection or settings.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 128) -> int:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )
        return len(chunks)

    def delete_source(self, source: str) -> None:
        """Remove every chunk of a document, so re-ingesting an updated file leaves no stale chunks."""
        self.collection.delete(where={"source": source})

    def search(self, query: str, k: int = 4, source: str | None = None) -> list[SearchHit]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count()),
            where={"source": source} if source else None,
        )
        return [
            SearchHit(text=doc, source=meta["source"], page=meta["page"], section=meta["section"], distance=dist)
            for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]

    def sources(self) -> list[str]:
        metadatas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metadatas})

    def count(self) -> int:
        return self.collection.count()
