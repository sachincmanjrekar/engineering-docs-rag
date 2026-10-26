"""Evaluate retrieval (and optionally answers) across chunk sizes.

Retrieval metrics (no API key needed):
  hit@1 / hit@k  - share of answerable questions whose expected page is in the top 1 / top k chunks
  MRR            - mean reciprocal rank of the first chunk from the expected page
  abstain        - share of unanswerable questions where every chunk was above the distance threshold
                   (only meaningful as a sanity check; the LLM's own "not found" rule is the main guard)

With --llm (needs ANTHROPIC_API_KEY), also measures:
  answer acc     - answerable: expected fact appears in the answer; unanswerable: answer is the "not found" reply

Usage: python eval/evaluate.py [--llm] [--chunk-sizes 256 512 1024]
"""
import argparse
import json
from pathlib import Path

from app.config import settings
from app.ingest import chunk_document
from app.llm import NOT_FOUND
from app.rag import RagPipeline
from app.vector_store import VectorStore

ROOT = Path(__file__).resolve().parent.parent
MANUAL = ROOT / "sample_docs" / "PT-200_Pressure_Transmitter_Manual.pdf"
EVAL_SET = ROOT / "eval" / "eval_set.json"
RESULTS = ROOT / "eval" / "results.md"


def evaluate(chunk_size: int, cases: list[dict], k: int, llm=None) -> dict:
    chunks = chunk_document(MANUAL.read_bytes(), MANUAL.name, chunk_size, chunk_size // 8)
    store = VectorStore(path=None, collection=f"eval_{chunk_size}")
    store.add_chunks(chunks)
    pipeline = RagPipeline(store, llm, top_k=k)

    answerable = [c for c in cases if c["expected_page"] is not None]
    unanswerable = [c for c in cases if c["expected_page"] is None]

    hit1 = hitk = rr = 0.0
    for case in answerable:
        pages = [h.page for h in store.search(case["question"], k=k)]
        if pages and pages[0] == case["expected_page"]:
            hit1 += 1
        if case["expected_page"] in pages:
            hitk += 1
            rr += 1 / (pages.index(case["expected_page"]) + 1)
    abstain = sum(not pipeline.retrieve(c["question"]) for c in unanswerable)

    row = {
        "chunk_size": chunk_size,
        "chunks": len(chunks),
        "hit@1": hit1 / len(answerable),
        f"hit@{k}": hitk / len(answerable),
        "MRR": rr / len(answerable),
        "abstain": abstain / len(unanswerable) if unanswerable else 0.0,
    }
    if llm:
        correct = 0
        for case in cases:
            answer = pipeline.ask(case["question"]).answer
            if case["expected_answer"] is None:
                correct += NOT_FOUND.lower().rstrip(".") in answer.lower()
            else:
                correct += case["expected_answer"].lower() in answer.lower()
        row["answer acc"] = correct / len(cases)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also score generated answers (calls the Claude API)")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--k", type=int, default=settings.top_k)
    args = parser.parse_args()

    if not MANUAL.exists():
        raise SystemExit("Sample manual missing - run: python scripts/make_sample_manual.py")
    cases = json.loads(EVAL_SET.read_text())
    llm = None
    if args.llm:
        from app.llm import ClaudeLLM

        llm = ClaudeLLM()

    rows = [evaluate(size, cases, args.k, llm) for size in args.chunk_sizes]
    headers = list(rows[0].keys())
    lines = [
        f"# Evaluation results\n\n{len(cases)} questions on `{MANUAL.name}`, top-k = {args.k}, "
        f"embedding model `{settings.embedding_model}`, distance threshold {settings.max_distance}.\n",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in rows:
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row.values()) + " |")
    report = "\n".join(lines) + "\n"
    RESULTS.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
