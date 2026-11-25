"""
src/index.py - Step 4 of DocLens: embed chunks and store them in ChromaDB.

Usage (from the doclens/ folder):
    python src/index.py

Reads  data/processed/chunks.jsonl
Writes a persistent ChromaDB database in data/chroma/

Every run rebuilds the collection from scratch, so it is safe to re-run
after changing the chunk settings.
"""
import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
DB_DIR = "data/chroma"
COLLECTION = "doclens"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH = 64

# bge models work best when SEARCH QUESTIONS (not documents) start with this.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"{CHUNKS_FILE} not found. Run src/chunk.py first.")
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading model {MODEL_NAME} (first run downloads about 130 MB)...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # collection did not exist yet
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    start = time.time()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        col.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]}
                for c in batch
            ],
        )
        print(f"  indexed {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print(f"\nDone in {time.time() - start:.0f}s. Stored {col.count()} chunks in {DB_DIR}")

    # Quick sanity check: search for something and show where it was found.
    question = "How do I file an income tax return?"
    q_emb = model.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=3)
    print(f"\nTest question: {question}")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"\n[{meta['source']} | page {meta['page']} | distance {dist:.3f}]")
        print(doc[:200].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()