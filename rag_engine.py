import json
import re
from functools import lru_cache
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings


# -----------------------------
# Load documents and BM25 once
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
GUIDELINES_PATH = BASE_DIR / "guidelines.json"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "health_guidelines"

with GUIDELINES_PATH.open("r", encoding="utf-8") as f:
    documents = json.load(f)

texts = [f"{d['title']} {d['topic']} {d['text']}" for d in documents]


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


bm25 = BM25Okapi([tokenize(text) for text in texts])

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def _embeddings_for(api_key):
    if not api_key:
        raise ValueError("OpenAI API key is required for retrieval.")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )


def _ensure_collection_indexed(api_key):
    """Index documents only when the persistent collection is empty."""
    if collection.count() > 0:
        return

    embeddings = _embeddings_for(api_key)
    vectors = embeddings.embed_documents(texts)
    collection.add(
        ids=[str(d["id"]) for d in documents],
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {
                "title": d["title"],
                "topic": d["topic"],
                "source": d["source"],
                "source_url": d["source_url"],
            }
            for d in documents
        ],
    )


def hybrid_search(query, top_k=3, api_key=None):
    """Combine BM25 lexical retrieval with Chroma vector retrieval using RRF."""
    if not api_key:
        raise ValueError("OpenAI API key is required for hybrid search.")

    # BM25 results
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:10]

    # Chroma collection is initialized lazily so the dashboard key is used.
    _ensure_collection_indexed(api_key)
    embeddings = _embeddings_for(api_key)
    query_vector = embeddings.embed_query(query)
    chroma_results = collection.query(query_embeddings=[query_vector], n_results=min(10, max(1, collection.count())))
    chroma_ids = chroma_results.get("ids", [[]])[0]

    # Reciprocal Rank Fusion
    fused_scores = {}
    for rank, i in enumerate(bm25_indices):
        doc_id = str(documents[i]["id"])
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (60 + rank + 1)
    for rank, doc_id in enumerate(chroma_ids):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (60 + rank + 1)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
    docs_by_id = {str(d["id"]): d for d in documents}
    return [docs_by_id[doc_id] for doc_id in ranked_ids if doc_id in docs_by_id]
