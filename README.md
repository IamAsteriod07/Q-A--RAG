# Tiny RAG Q&A Service

This repository contains a small Q&A service over a tiny set of industrial safety PDFs.

Files added:

- `indexer.py` — builds a SQLite DB of chunks and a FAISS index using a local embedding model.
- `service.py` — FastAPI app exposing POST /ask for querying. Modes: `baseline` (FAISS cosine) and `hybrid` (vector + keyword blending).

Quick start

1. Install requirements (prefer a virtualenv):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Build the index from the `industrial/` folder (make sure the PDFs are present):

```powershell
python indexer.py
```

3. Run the service:

```powershell
uvicorn service:app --reload --port 8000
```

4. Example request:

POST http://localhost:8000/ask

Body (JSON):

```json
{ "q": "What are the minimum safety distances for robots?", "k": 5, "mode": "hybrid" }
```

The response includes `answer` (short grounded text or null), `contexts` with scores and `reranker_used`.

Notes and next steps

- The `hybrid` reranker is a simple blend: alpha*vector + (1-alpha)*keyword_overlap.
- A learned reranker (logistic regression) or BM25/SQLite FTS can be added for better results.
- For answers, this toy service returns the best-matching chunk. Integrate an LLM to generate concise grounded answers from the retrieved contexts.
