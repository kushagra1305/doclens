"""
src/ingest.py - Step 2 of DocLens: parse PDFs into clean, page-level records.

Usage (from the doclens/ folder):
    python src/ingest.py

Reads every PDF in data/raw/ and writes data/processed/pages.jsonl,
one JSON object per page:
    {"source": "file.pdf", "page": 3, "text": "..."}

Page numbers are 1-based so they match what a human sees in a PDF viewer.
Keeping source + page on every record is what makes citations possible later.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/pages.jsonl")
MIN_CHARS = 40          # pages shorter than this are treated as empty/scanned
REPEAT_THRESHOLD = 0.5  # a line on >50% of pages is a header/footer


def extract_pages(pdf_path: Path) -> list[str]:
    """Return raw text for each page of a PDF."""
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text("text") for page in doc]


def _key(line: str) -> str:
    """Compare lines with digits masked, so '10:41 am' and '10:50 am' match."""
    return re.sub(r"\d+", "#", line.strip())


def find_repeated_lines(pages: list[str]) -> set[str]:
    """Lines that repeat across most pages are headers/footers; find them."""
    if len(pages) < 4:
        return set()
    counts = Counter()
    for text in pages:
        # only the first/last 5 lines of each page are header/footer candidates
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for k in set(_key(ln) for ln in lines[:5] + lines[-5:]):
            counts[k] += 1
    cutoff = len(pages) * REPEAT_THRESHOLD
    return {k for k, c in counts.items() if c >= cutoff}


def clean_text(text: str, repeated: set[str]) -> str:
    """Remove headers/footers, fix hyphenation, normalise whitespace."""
    lines = [ln for ln in text.splitlines() if _key(ln) not in repeated]
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)       # re-join hyphenated words
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # single newline -> space
    text = re.sub(r"[ \t]+", " ", text)                # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)             # collapse blank lines
    return text.strip()


def main() -> None:
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {RAW_DIR.resolve()}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0

    with OUT_FILE.open("w", encoding="utf-8") as out:
        for pdf in pdfs:
            try:
                raw_pages = extract_pages(pdf)
            except Exception as exc:  # corrupt/encrypted file
                print(f"[skip] {pdf.name}: {exc}")
                continue

            repeated = find_repeated_lines(raw_pages)
            for i, raw in enumerate(raw_pages, start=1):
                text = clean_text(raw, repeated)
                if len(text) < MIN_CHARS:
                    skipped += 1
                    continue
                record = {"source": pdf.name, "page": i, "text": text}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            print(f"[ok] {pdf.name}: {len(raw_pages)} pages")

    print(f"\nDone. Kept {kept} pages, skipped {skipped} near-empty pages.")
    print(f"Output: {OUT_FILE}")
    if skipped > kept * 0.2:
        print("Warning: many empty pages - some PDFs may be scanned (need OCR).")


if __name__ == "__main__":
    main()