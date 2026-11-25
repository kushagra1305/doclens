"""
src/chunk.py - Step 3 of DocLens: split pages into overlapping chunks.

Usage (from the doclens/ folder):
    python src/chunk.py
    python src/chunk.py --size 600 --overlap 100   # for experiments later

Reads  data/processed/pages.jsonl
Writes data/processed/chunks.jsonl, one JSON object per chunk:
    {"id": "file.pdf::p3::c0", "source": "file.pdf", "page": 3,
     "chunk_index": 0, "text": "..."}

Chunks never cross page boundaries, so every chunk has one exact page
number to cite.
"""
import argparse
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

IN_FILE = Path("data/processed/pages.jsonl")
OUT_FILE = Path("data/processed/chunks.jsonl")
MIN_CHUNK_CHARS = 50  # drop tiny leftovers like stray page numbers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000, help="max characters per chunk")
    parser.add_argument("--overlap", type=int, default=150, help="characters shared between neighbours")
    args = parser.parse_args()

    if not IN_FILE.exists():
        raise SystemExit(f"{IN_FILE} not found. Run src/ingest.py first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.size,
        chunk_overlap=args.overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    n_pages = n_chunks = 0
    with IN_FILE.open(encoding="utf-8") as fin, OUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            page = json.loads(line)
            n_pages += 1
            pieces = [p for p in splitter.split_text(page["text"]) if len(p) >= MIN_CHUNK_CHARS]
            for i, text in enumerate(pieces):
                record = {
                    "id": f"{page['source']}::p{page['page']}::c{i}",
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": i,
                    "text": text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks += 1

    avg = n_chunks / n_pages if n_pages else 0
    print(f"Pages read:    {n_pages}")
    print(f"Chunks saved:  {n_chunks}  (about {avg:.1f} per page)")
    print(f"Chunk size {args.size}, overlap {args.overlap}")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()