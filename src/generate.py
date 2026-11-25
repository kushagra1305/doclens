"""
src/generate.py - Steps 5 and 8 of DocLens: answer a question with citations.

Usage (from the doclens/ folder):
    python src/generate.py "your question here"

Flow:
  1. Fetch 30 candidate chunks with vector search and re-score them with the
     cross-encoder reranker; keep the best 5.
  2. "I don't know" gate: if the best reranker score is below RERANK_CUTOFF,
     refuse WITHOUT calling the language model.
  3. Otherwise ask the local Ollama model to answer ONLY from those passages,
     citing passage numbers.
  4. Print the answer plus the unique file + page sources.
"""
from __future__ import annotations

import argparse

import requests

from retrieve import search

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"
TOP_K = 5
RETRIEVAL_MODE = "vecrerank"

# Refuse when the best reranker score is below this. From the evaluation:
# -0.8 refused 67% of unanswerable questions (6% wrong refusals);
# 1.6 refused 100% (11% wrong refusals). The language model is a second guard
# (it says "I don't know" when passages do not answer), so the gate can be
# lenient: at 0.0 it wrongly refused a good question, so we start at -2.0.
RERANK_CUTOFF = -2.0

IDK = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered passages provided.
Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- Cite only the passages that directly support what you wrote, like [1] or [2][3]. Do not cite a passage that is about a different section or topic than the question.
- If the passages do not contain the answer, reply exactly: {IDK}
- Keep the answer short and in plain language."""


def build_context(hits: list[dict]) -> str:
    blocks = []
    for n, h in enumerate(hits, start=1):
        blocks.append(f"[{n}] ({h['source']}, page {h['page']})\n{h['text']}")
    return "\n\n".join(blocks)


def call_llm(system: str, user: str, model: str | None = None) -> str:
    """Send one request to the local Ollama server and return the reply text."""
    model = model or MODEL
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit("Cannot reach Ollama. Is it installed and running? Try: ollama run " + model)
    return r.json()["message"]["content"].strip()


def unique_sources(hits: list[dict], numbers: list[int]) -> list[dict]:
    """One entry per (file, page), listing every passage number that pointed to it."""
    merged: dict[tuple, dict] = {}
    for n in numbers:
        h = hits[n - 1]
        key = (h["source"], h["page"])
        merged.setdefault(key, {"source": h["source"], "page": h["page"], "passages": []})
        merged[key]["passages"].append(n)
    return list(merged.values())


def answer(question: str, model: str | None = None) -> dict:
    hits = search(question, TOP_K, RETRIEVAL_MODE)
    top_score = hits[0]["rerank_score"] if hits else None

    if not hits or top_score < RERANK_CUTOFF:
        return {"answer": IDK, "sources": [], "cited": True, "hits": hits,
                "top_score": top_score, "refused_by": "score_gate"}

    user_msg = f"Passages:\n\n{build_context(hits)}\n\nQuestion: {question}"
    text = call_llm(SYSTEM_PROMPT, user_msg, model)

    if IDK.lower() in text.lower():
        return {"answer": IDK, "sources": [], "cited": True, "hits": hits,
                "top_score": top_score, "refused_by": "model"}

    cited_numbers = [n for n in range(1, len(hits) + 1) if f"[{n}]" in text]
    cited = bool(cited_numbers)
    # If the model forgot to cite, fall back to the top passages and say so.
    numbers = cited_numbers if cited else [1, 2, 3][: len(hits)]
    return {"answer": text, "sources": unique_sources(hits, numbers), "cited": cited,
            "hits": hits, "top_score": top_score, "refused_by": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    result = answer(args.question)
    print(f"\nQuestion: {args.question}\n")
    print(f"Answer: {result['answer']}\n")
    if result["sources"]:
        label = "Sources:" if result["cited"] else "Sources (closest passages; the model did not cite):"
        print(label)
        for s in result["sources"]:
            nums = ", ".join(f"[{n}]" for n in s["passages"])
            print(f"  {nums} {s['source']}, page {s['page']}")
    if result["top_score"] is not None:
        why = f", refused by {result['refused_by']}" if result["refused_by"] else ""
        print(f"\n(best reranker score: {result['top_score']:.2f}; cutoff {RERANK_CUTOFF}{why})")


if __name__ == "__main__":
    main()