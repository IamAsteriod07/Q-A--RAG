import os
import sqlite3
import faiss
import pickle
import json
import math
from typing import List, Tuple
import numpy as np
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch


DB_PATH = "chunks.db"
FAISS_INDEX_PATH = "faiss.index"
ID_MAP_PATH = "id_map.pkl"


def load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_texts(texts: List[str], tokenizer, model, device='cpu') -> np.ndarray:
    # Compute sentence embeddings with mean pooling
    batch_size = 16
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = summed / summed_mask
            emb = mean_pooled.cpu().numpy()
            all_embs.append(emb)
    return np.vstack(all_embs)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    text = text.replace('\r', '\n')
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                end = start + chunk_size
                chunks.append(p[start:end].strip())
                start = end - overlap
                if start < 0:
                    start = 0
    return chunks


def ensure_db(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id INTEGER PRIMARY KEY,
        title TEXT,
        source TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY,
        doc_id INTEGER,
        text TEXT,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    )
    """)
    con.commit()
    return con


def build_index(pdf_folder: str = "./industrial", rebuild: bool = False):
    # If index exists and not rebuild, skip
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DB_PATH) and os.path.exists(ID_MAP_PATH) and not rebuild:
        print("Index already exists. Use rebuild=True to recreate.")
        return

    tokenizer, model = load_model()

    con = ensure_db()
    cur = con.cursor()

    # Clear tables
    cur.execute("DELETE FROM chunks")
    cur.execute("DELETE FROM documents")
    con.commit()

    chunk_texts = []
    chunk_meta = []  # tuples (doc_id, title, source)

    doc_id = 0
    for fname in sorted(os.listdir(pdf_folder)):
        if not fname.lower().endswith('.pdf'):
            continue
        doc_id += 1
        path = os.path.join(pdf_folder, fname)
        title = fname
        source = f"file://{os.path.abspath(path)}"
        cur.execute("INSERT INTO documents (doc_id, title, source) VALUES (?, ?, ?)", (doc_id, title, source))

        reader = PdfReader(path)
        full_text = []
        for page in reader.pages:
            try:
                full_text.append(page.extract_text() or "")
            except Exception:
                full_text.append("")
        text = "\n\n".join(full_text)
        chunks = chunk_text(text)
        for c in chunks:
            chunk_texts.append(c)
            chunk_meta.append((doc_id, title, source))

    # Embed all chunks
    print(f"Embedding {len(chunk_texts)} chunks...")
    embeddings = embed_texts(chunk_texts, tokenizer, model)

    # Normalize for cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save metadata and id map
    id_map = []
    for i, (doc_id_, title, source) in enumerate(chunk_meta, start=1):
        # chunk ids will be 1-based
        cur.execute("INSERT INTO chunks (chunk_id, doc_id, text) VALUES (?, ?, ?)", (i, doc_id_, chunk_texts[i-1]))
        id_map.append(i)

    con.commit()
    con.close()

    with open(ID_MAP_PATH, 'wb') as f:
        pickle.dump({'id_map': id_map}, f)

    print("Index built and saved.")


if __name__ == '__main__':
    build_index()
