<<<<<<< HEAD
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
=======
# A lightweight Retrieval-Augmented Generation (RAG) question-answering project that indexes documents, retrieves relevant chunks via vector search, and generates grounded answers using an LLM with cited sources.

 # 🔎Features
Document ingestion and chunking for PDFs, text, and URLs.

Embedding generation and vector search using FAISS.

RAG pipeline for context retrieval + answer generation with citations.

Optional API and UI for interactive Q&A (FastAPI/Streamlit).

# 🧠Architecture
Ingestion: Load documents, split into chunks, and create embeddings.

Index: Store embeddings in a vector database (e.g., Chroma).

Retrieval: Hybrid or vector similarity search to find top-k relevant chunks.

Generation: Prompt an LLM with question + retrieved context to produce answers with sources.

# 🛠️Tech Stack
Language: Python 3.10+

LLM: Google Gemini, or local models (configurable).

Embeddings: all-MiniLM-L6-v2 or OpenAI embeddings.

Vector DB: FAISS and ChromaDB (default).

API/UI: FastAPI and/or Streamlit .

 # 🖥️Project Structure
src/ingest/: loaders and chunkers for PDFs/URLs/files.

src/index/: embedding and vector store initialization.

src/retrieve/: retriever utilities (similarity, reranking).

src/generate/: prompt templates and answer synthesis.

api/: FastAPI routes for upload, index, and query.

app/: Streamlit UI .

# Getting Started
Prerequisites

Python 3.10+ and pip/uv.

Optional: Docker and Docker Compose.

API keys (choose one):

GROQ API key (GR0Q_API_KEY).

Google Gemini key (GEMINI_API_KEY).

# Clone

git clone https://github.com/IamAsteriod07/Q-A--RAG && cd Q-A--RAG.

# Environment
Copy .env.example to .env and set required keys:

OPENAI_API_KEY or GEMINI_API_KEY
VECTOR_STORE=chroma|faiss

EMBEDDING_MODEL=all-MiniLM-L6-v2|text-embedding-3-small

DATA_DIR=./data

INDEX_DIR=./index

# 📦Installation
pip install -r requirements.txt

# Troubleshooting
Empty answers: check embeddings model and index path, rebuild index.

Slow responses: reduce chunk size, top_k, or switch to smaller embedding model.

Missing citations: ensure retriever returns source metadata and prompt includes citations.

Or with uv: uv pip install -r requirements.txt.



>>>>>>> 052ea3f816506b8dd569f8f5926e57a30c73ce20
