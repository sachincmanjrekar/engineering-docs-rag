# Engineering Document Q&A Assistant (RAG)

Ask questions about technical PDFs such as product manuals, datasheets and specifications, and get answers that cite their sources, built on **Retrieval-Augmented Generation (RAG)**.

Engineers lose a lot of time searching long manuals for one torque value, error code or calibration step. This service ingests the PDFs, retrieves the passages that matter, and has an LLM answer **only from those passages**, with page-level citations. When the documents don't contain the answer, it says *"Not found in the provided documents."* instead of guessing.

## Architecture

```
            INGESTION                                    QUERY
 PDF ──► PyMuPDF text extraction            Question ──► embed (Sentence-Transformers)
     ──► cleaning (whitespace, hyphenation)          ──► ChromaDB top-k cosine search
     ──► chunking with overlap                       ──► drop chunks above distance threshold
         (LangChain RecursiveCharacterTextSplitter)  ──► nothing left? → "Not found"
     ──► metadata: file, page, section heading       ──► grounded prompt with numbered passages
     ──► embeddings (all-MiniLM-L6-v2)               ──► Claude answers with [1][2] citations
     ──► ChromaDB (persistent)                       ──► answer + source pages returned
```

| Layer | Tech |
|---|---|
| PDF extraction | PyMuPDF |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (size/overlap configurable) |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` (384-dim, runs locally on CPU) |
| Vector DB | ChromaDB, cosine distance, metadata filtering by document |
| LLM | Claude via the Anthropic Python SDK |
| API | FastAPI (`/ingest`, `/ask`, `/documents`, `/health`) |
| UI | Streamlit chat with expandable source passages |
| Quality | Pytest unit/API tests (offline, LLM mocked) + retrieval evaluation script |

## Project structure

```
app/
  config.py        settings from environment variables
  ingest.py        PDF -> clean text -> chunks with page/section metadata
  vector_store.py  ChromaDB wrapper (upsert, delete by document, top-k search)
  llm.py           grounded system prompt + Claude call
  rag.py           retrieve -> filter -> generate pipeline
  api.py           FastAPI endpoints
ui/streamlit_app.py
eval/              eval_set.json (Q&A pairs), evaluate.py, results.md
scripts/make_sample_manual.py   builds a sample 6-page transmitter manual (fictional product)
tests/             pytest suite
```

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                              # then put your ANTHROPIC_API_KEY in .env

python scripts/make_sample_manual.py                # creates sample_docs/PT-200_...pdf
uvicorn app.api:app --reload                        # API docs at http://localhost:8000/docs
streamlit run ui/streamlit_app.py                   # chat UI at http://localhost:8501
```

Or use the API directly:

```bash
curl -F "file=@sample_docs/PT-200_Pressure_Transmitter_Manual.pdf" http://localhost:8000/ingest
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d "{\"question\": \"What torque should the process connection be tightened to?\"}"
```

Docker:

```bash
docker build -t docs-rag .
docker run -p 8000:8000 --env-file .env docs-rag
```

## Evaluation

`eval/eval_set.json` holds 18 questions about the sample manual: 16 with a known answer page and fact, and 2 that the manual can't answer (price, Bluetooth). `eval/evaluate.py` rebuilds the index for several chunk sizes and measures:

- **hit@1 / hit@k:** how often the correct page is the top result, or within the top k results
- **MRR:** mean reciprocal rank of the first correct chunk
- **abstain:** how often unanswerable questions retrieve nothing above the relevance threshold
- **answer acc** (`--llm` only): whether the expected fact appears in the answer, or "Not found" for unanswerable questions

```bash
python eval/evaluate.py            # retrieval only, no API key needed
python eval/evaluate.py --llm      # also scores generated answers
```

Retrieval results on the sample manual:

18 questions on `PT-200_Pressure_Transmitter_Manual.pdf`, top-k = 4, embedding model `all-MiniLM-L6-v2`, distance threshold 0.75.

| chunk_size | chunks | hit@1 | hit@4 | MRR | abstain |
|---|---|---|---|---|---|
| 256 | 24 | 0.94 | 1.00 | 0.95 | 0.00 |
| 512 | 12 | 0.81 | 1.00 | 0.88 | 0.00 |
| 1024 | 6 | 0.81 | 1.00 | 0.86 | 0.00 |

**What the numbers show:**
- Every answer page is in the top 4 at every chunk size (hit@4 = 1.00). The difference is ranking.
- **256-character chunks rank best** (hit@1 0.94, MRR 0.95). Small chunks hold one spec or step each, so the question embedding matches them more precisely. Larger chunks mix several topics and dilute the match. I made 256 (overlap 32) the default because of this.
- **The distance threshold doesn't separate unanswerable questions** (abstain 0.00). "What is the price of the PT-200?" sits at cosine distance 0.45 because it names the product, while some answerable questions score up to 0.67. The threshold only filters clearly off-topic questions. The real protection is the grounded prompt, which must answer "Not found" when the passages don't contain the fact. `--llm` measures that.
- The answer-level score (`--llm`) needs an API key and wasn't run for this table.
- Caveat: one short manual and 18 questions are enough to compare settings, but not enough to claim production accuracy.

## Tests

```bash
pytest -q
```

The tests run offline. They use a deterministic hash embedding and a fake LLM. They cover text cleaning, section detection, chunk metadata, the ingest → ask API flow, re-ingest de-duplication, input validation, and the rule that the LLM is never called when nothing relevant is retrieved.

## Design decisions

- **Overlap between chunks** keeps a sentence that straddles a chunk boundary retrievable from both sides.
- **Section and page metadata** on each chunk lets answers cite *where* a fact came from, so an engineer can verify it.
- **Two guards against hallucination:** a distance threshold drops irrelevant chunks before the LLM is called, and the system prompt tells the model to answer only from the numbered passages and to reply "Not found" otherwise.
- **Re-ingesting a file deletes its old chunks first**, so an updated manual never mixes old and new values.
- **Local embeddings** mean document text is sent to the LLM only at question time, and only the retrieved passages.

## Limitations and next steps

- Scanned PDFs need OCR (e.g. Tesseract) before ingestion.
- Tables are flattened to text. Table-aware parsing would improve spec-sheet lookups.
- Hybrid search (BM25 + vectors) and a cross-encoder re-ranker would help with exact part numbers and error codes.
- Evaluate on a larger, real document set, and add LLM-as-judge faithfulness scoring.
