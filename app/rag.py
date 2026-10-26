"""Retrieval-Augmented Generation pipeline: retrieve -> filter -> generate -> cite."""
from dataclasses import asdict, dataclass

from app.config import settings
from app.llm import NOT_FOUND
from app.vector_store import SearchHit, VectorStore


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict]


class RagPipeline:
    def __init__(self, store: VectorStore, llm, top_k: int | None = None, max_distance: float | None = None):
        self.store = store
        self.llm = llm
        self.top_k = top_k or settings.top_k
        self.max_distance = settings.max_distance if max_distance is None else max_distance

    def retrieve(self, question: str, source: str | None = None) -> list[SearchHit]:
        hits = self.store.search(question, k=self.top_k, source=source)
        return [h for h in hits if h.distance <= self.max_distance]

    def ask(self, question: str, source: str | None = None) -> RagAnswer:
        hits = self.retrieve(question, source)
        if not hits:
            # Nothing relevant retrieved: refuse instead of letting the LLM guess.
            return RagAnswer(answer=NOT_FOUND, sources=[])
        answer = self.llm.answer(question, hits)
        return RagAnswer(answer=answer, sources=[asdict(h) for h in hits])
