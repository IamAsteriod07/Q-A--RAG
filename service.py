import os
import faiss
import pickle
import sqlite3
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
from indexer import FAISS_INDEX_PATH, ID_MAP_PATH, DB_PATH, load_model, embed_texts


app = FastAPI(title="Tiny RAG Q&A")


class AskRequest(BaseModel):
    q: str
    k: int = 5
    mode: str = "baseline"  # baseline | hybrid


def load_resources():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise RuntimeError("FAISS index not found; run indexer.build_index first")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(ID_MAP_PATH, 'rb') as f:
        id_map = pickle.load(f)['id_map']
    tokenizer, model = load_model()
    return index, id_map, tokenizer, model


def get_db_connection():
    return sqlite3.connect(DB_PATH)


@app.on_event("startup")
def startup():
    global INDEX, ID_MAP, TOKENIZER, MODEL
    INDEX, ID_MAP, TOKENIZER, MODEL = load_resources()


def search_faiss(query: str, k: int):
    emb = embed_texts([query], TOKENIZER, MODEL)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-9, None)
    D, I = INDEX.search(emb.astype('float32'), k)
    return D[0].tolist(), I[0].tolist()


def keyword_score(query: str, text: str) -> float:
    q_tokens = set([t.lower() for t in query.split() if len(t) > 2])
    t_tokens = set([t.lower().strip('.,') for t in text.split() if len(t) > 2])
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def rerank_hybrid(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Simple blending: final = alpha * vector_score + (1-alpha) * keyword_score
    alpha = 0.7
    for c in candidates:
        ks = keyword_score(query, c['text'])
        vs = c.get('vector_score', 0.0)
        c['rerank_score'] = alpha * vs + (1 - alpha) * ks
        c['keyword_score'] = ks
    return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)


@app.post('/ask')
def ask(req: AskRequest):
    try:
        D, I = search_faiss(req.q, req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Map results to texts
    con = get_db_connection()
    cur = con.cursor()
    contexts = []
    for score, idx in zip(D, I):
        # faiss returns 0-based indices
        chunk_id = idx + 1
        cur.execute('SELECT text, doc_id FROM chunks WHERE chunk_id = ?', (chunk_id,))
        row = cur.fetchone()
        if not row:
            continue
        text, doc_id = row
        cur.execute('SELECT title, source FROM documents WHERE doc_id = ?', (doc_id,))
        meta = cur.fetchone() or (None, None)
        title, source = meta
        contexts.append({
            'chunk_id': chunk_id,
            'text': text,
            'doc_title': title,
            'source': source,
            'vector_score': float(score)
        })
    con.close()

    reranker_used = None
    if req.mode == 'baseline':
        sorted_ctx = sorted(contexts, key=lambda x: x['vector_score'], reverse=True)
        reranker_used = 'none'
    else:
        sorted_ctx = rerank_hybrid(req.q, contexts)
        reranker_used = 'hybrid'

    # Build a simple grounded answer: if top context has low score, abstain
    top = sorted_ctx[0] if sorted_ctx else None
    if not top or top['vector_score'] < 0.05:
        answer = None
        note = "Insufficient evidence to answer confidently from the documents."
    else:
        # For brevity return a short answer by echoing the top context (in real system use an LLM)
        snippet = top['text']
        # trim to ~300 chars
        answer = snippet.strip()[:500]
        note = None

    return {
        'answer': answer,
        'note': note,
        'contexts': sorted_ctx,
        'reranker_used': reranker_used
    }
