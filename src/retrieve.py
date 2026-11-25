"""
src/retrieve.py - find the chunks most relevant to a question.

Modes:
  vector    : meaning-based search with embeddings (ChromaDB)            [Step 5]
  hybrid    : vector + BM25 keyword search merged with RRF               [Step 7]
  rerank    : hybrid candidates re-scored by a cross-encoder             [Step 7]
  vecrerank : vector-only candidates re-scored by a cross-encoder        [Step 7]

Usage (from the doclens/ folder):
    python src/retrieve.py "your question here"
    python src/retrieve.py "your question here" --k 8 --mode rerank

Other files import the retrieve_* functions from here.

IMPORTANT: hybrid mode reads data/processed/chunks.jsonl for BM25, so that file
must match the chunks stored in the database. If you change chunk settings,
re-run chunk.py and then index.py.
"""
import argparse
import json
import re
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DB_DIR = "data/chroma"
COLLECTION = "doclens"
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # standard constant for Reciprocal Rank Fusion

STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it its of on or "
    "that the this to under what when which who will with my me".split()
)

_model = None
_collection = None
_bm25 = None
_chunks = None
_reranker = None


def _load():
    """Load the embedding model and database once, then reuse them."""
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        _collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    return _model, _collection


def _tokenize(text: str) -> list[str]:
    """Lowercase words and numbers (so '248' and '194N' stay searchable)."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def _load_bm25():
    """Build the BM25 keyword index from chunks.jsonl once, then reuse it."""
    global _bm25, _chunks
    if _bm25 is None:
        with CHUNKS_FILE.open(encoding="utf-8") as f:
            _chunks = [json.loads(line) for line in f]
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _chunks])
    return _bm25, _chunks


def retrieve(question: str, k: int = 5) -> list[dict]:
    """Vector search: the k closest chunks, best first."""
    model, col = _load()
    emb = model.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=emb, n_results=k)
    hits = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {"id": cid, "text": doc, "source": meta["source"], "page": meta["page"], "distance": dist}
        )
    return hits


def retrieve_hybrid(question: str, k: int = 5, pool: int = 30) -> list[dict]:
    """Hybrid search: merge vector and BM25 rankings with RRF.

    'distance' is the vector distance when the chunk was in the vector results,
    otherwise None.
    """
    vec_hits = retrieve(question, pool)
    bm25, chunks = _load_bm25()
    scores = bm25.get_scores(_tokenize(question))
    top_idx = np.argsort(scores)[::-1][:pool]

    fused: dict[str, float] = {}
    info: dict[str, dict] = {}

    for rank, h in enumerate(vec_hits, start=1):
        fused[h["id"]] = fused.get(h["id"], 0.0) + 1.0 / (RRF_K + rank)
        info[h["id"]] = h

    for rank, i in enumerate(top_idx, start=1):
        if scores[i] <= 0:
            continue
        c = chunks[int(i)]
        fused[c["id"]] = fused.get(c["id"], 0.0) + 1.0 / (RRF_K + rank)
        if c["id"] not in info:
            info[c["id"]] = {
                "id": c["id"], "text": c["text"], "source": c["source"],
                "page": c["page"], "distance": None,
            }

    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [info[cid] for cid in best]


def _load_reranker():
    """Load the cross-encoder once (first run downloads about 90 MB)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    return _reranker


def rerank(question: str, candidates: list[dict], k: int = 5) -> list[dict]:
    """Re-score candidates by reading question + passage together; keep the best k.

    Adds a 'rerank_score' to each hit (higher = more relevant).
    """
    if not candidates:
        return []
    scores = _load_reranker().predict([(question, c["text"]) for c in candidates])
    order = np.argsort(scores)[::-1][:k]
    out = []
    for i in order:
        hit = dict(candidates[int(i)])
        hit["rerank_score"] = float(scores[int(i)])
        out.append(hit)
    return out


def retrieve_reranked(question: str, k: int = 5, candidates: int = 30, hybrid: bool = True) -> list[dict]:
    """Fetch a wide pool of candidates, then let the cross-encoder pick the best k."""
    if hybrid:
        pool = retrieve_hybrid(question, k=candidates, pool=candidates)
    else:
        pool = retrieve(question, candidates)
    return rerank(question, pool, k)


def search(question: str, k: int = 5, mode: str = "vector") -> list[dict]:
    """One entry point for every mode."""
    if mode == "hybrid":
        return retrieve_hybrid(question, k)
    if mode == "rerank":
        return retrieve_reranked(question, k, hybrid=True)
    if mode == "vecrerank":
        return retrieve_reranked(question, k, hybrid=False)
    return retrieve(question, k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["vector", "hybrid", "rerank", "vecrerank"], default="vector")
    args = parser.parse_args()

    print(f"Question: {args.question}   (mode: {args.mode})\n")
    for rank, h in enumerate(search(args.question, args.k, args.mode), start=1):
        dist = f"{h['distance']:.3f}" if h["distance"] is not None else "n/a"
        extra = f" | rerank score {h['rerank_score']:.3f}" if "rerank_score" in h else ""
        print(f"#{rank}  {h['source']} | page {h['page']} | distance {dist}{extra}")
        print("    " + h["text"][:300].replace("\n", " ") + "...\n")


if __name__ == "__main__":
    main()