"""Grounded answer generation with Claude (Anthropic SDK)."""
import anthropic

from app.config import settings
from app.vector_store import SearchHit

NOT_FOUND = "Not found in the provided documents."

SYSTEM_PROMPT = f"""You answer questions about engineering documents (manuals, datasheets, specifications).

Rules:
- Use ONLY the numbered context passages supplied by the user. Do not use outside knowledge.
- Cite every fact with its passage number in square brackets, e.g. [1] or [2][3].
- Quote numbers, units, part numbers and limits exactly as written in the context.
- If the context does not contain the answer, reply exactly: "{NOT_FOUND}"
- Be concise: a short answer first, then supporting detail if useful."""


def build_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h.source}, page {h.page}, section: {h.section})\n{h.text}"
        for i, h in enumerate(hits, start=1)
    )


def build_user_message(question: str, hits: list[SearchHit]) -> str:
    return f"<context>\n{build_context(hits)}\n</context>\n\nQuestion: {question}"


class ClaudeLLM:
    def __init__(self, model: str | None = None):
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or settings.llm_model

    def answer(self, question: str, hits: list[SearchHit]) -> str:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(question, hits)}],
            output_config={"effort": "low"},  # short, grounded answers don't need deep reasoning
            # If a safety classifier declines, let the API retry on a suitable fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return "The model declined to answer this question."
        return "".join(block.text for block in response.content if block.type == "text").strip()
