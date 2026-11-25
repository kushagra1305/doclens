"""
src/evaluate.py - Step 6 of DocLens: measure how well retrieval and the
"I don't know" rule work, using eval/questions.json.

Usage (from the doclens/ folder):
    python src/evaluate.py
    python src/evaluate.py --label chunk600      # name this run for later comparison
    python src/evaluate.py --mode hybrid --label hybrid   # vector + BM25 search
    python src/evaluate.py --mode rerank --label rerank   # hybrid + cross-encoder
    python src/evaluate.py --mode vecrerank --label vecrerank   # vector + cross-encoder

What it reports (no language model is called, so it is fast and free):
  1. Retrieval hit rate: for answerable questions, is a chunk from the correct
     file AND page in the top k results?  (hit@1, hit@3, hit@5, hit@10, MRR)
  2. The "I don't know" trade-off: for different distance cutoffs, how many
     unanswerable questions are correctly refused, and how many answerable
     questions are wrongly refused?
  3. The list of questions that missed, so you can see what to improve.
  4. In rerank modes: how well the reranker's score separates answerable from
     unanswerable questions (a better "I don't know" signal than distance).

Results are saved to eval/results_<label>.json so you can compare runs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from retrieve import retrieve, search

QUESTIONS_FILE = Path("eval/questions.json")
CUTOFFS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def first_hit_rank(hits: list[dict], q: dict):
    """Rank (1-based) of the first hit from the right file and page, or None."""
    for rank, h in enumerate(hits, start=1):
        if h["source"] == q["source"] and h["page"] in q["pages"]:
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--mode", choices=["vector", "hybrid", "rerank", "vecrerank"], default="vector")
    args = parser.parse_args()

    if not QUESTIONS_FILE.exists():
        raise SystemExit(f"{QUESTIONS_FILE} not found.")
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    results = []
    for q in questions:
        vec_hits = retrieve(q["question"], args.k)
        # the "I don't know" signal is always the best VECTOR distance
        top_dist = vec_hits[0]["distance"] if vec_hits else 1.0
        hits = vec_hits if args.mode == "vector" else search(q["question"], args.k, args.mode)
        rank = first_hit_rank(hits, q) if q["answerable"] else None
        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answerable": q["answerable"],
                "expected": f"{q['source']} p{q['pages']}" if q["answerable"] else None,
                "rank": rank,
                "top_distance": round(top_dist, 3),
                "top_rerank_score": round(hits[0]["rerank_score"], 3)
                if hits and "rerank_score" in hits[0] else None,
                "top3": [
                    (h["source"][:25], h["page"],
                     round(h["distance"], 3) if h["distance"] is not None else None)
                    for h in hits[:3]
                ],
            }
        )

    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]

    print(f"\n=== Retrieval quality ({len(ans)} answerable questions, mode: {args.mode}) ===")
    for k in (1, 3, 5, 10):
        if k > args.k:
            continue
        hit = sum(1 for r in ans if r["rank"] is not None and r["rank"] <= k)
        print(f"  hit@{k:<2}: {hit}/{len(ans)}  ({100 * hit / len(ans):.0f}%)")
    mrr = sum(1 / r["rank"] for r in ans if r["rank"]) / len(ans)
    print(f"  MRR   : {mrr:.3f}   (1.0 = right page always ranked first)")

    print("\n=== 'I don't know' cutoff trade-off ===")
    print("  cutoff | unanswerable refused | answerable wrongly refused")
    for c in CUTOFFS:
        refused = sum(1 for r in unans if r["top_distance"] > c)
        wrong = sum(1 for r in ans if r["top_distance"] > c)
        print(f"  {c:.2f}   | {refused:>2}/{len(unans)} ({100 * refused / len(unans):>3.0f}%)"
              f"       | {wrong:>2}/{len(ans)} ({100 * wrong / len(ans):>3.0f}%)")

    print("\n=== Answerable questions whose right page was NOT in the top "
          f"{args.k} ===")
    misses = [r for r in ans if r["rank"] is None]
    if not misses:
        print("  none")
    for r in misses:
        print(f"  {r['id']}: {r['question']}")
        print(f"        expected {r['expected']}")
        print(f"        got      {r['top3'][0] if r['top3'] else '-'}")

    print("\n=== Closest distances (to help pick the cutoff) ===")
    a = sorted(r["top_distance"] for r in ans)
    u = sorted(r["top_distance"] for r in unans)
    print(f"  answerable   : min {a[0]:.3f}  median {a[len(a) // 2]:.3f}  max {a[-1]:.3f}")
    print(f"  unanswerable : min {u[0]:.3f}  median {u[len(u) // 2]:.3f}  max {u[-1]:.3f}")

    if all(r["top_rerank_score"] is not None for r in results):
        sa = sorted(r["top_rerank_score"] for r in ans)
        su = sorted(r["top_rerank_score"] for r in unans)
        print("\n=== Reranker score as the 'I don't know' signal (higher = more relevant) ===")
        print(f"  answerable   : min {sa[0]:.3f}  median {sa[len(sa) // 2]:.3f}  max {sa[-1]:.3f}")
        print(f"  unanswerable : min {su[0]:.3f}  median {su[len(su) // 2]:.3f}  max {su[-1]:.3f}")
        print("  refuse if score below | unanswerable refused | answerable wrongly refused")
        for t in np.linspace(min(sa[0], su[0]), max(sa[-1], su[-1]), 9)[1:-1]:
            refused = sum(1 for r in unans if r["top_rerank_score"] < t)
            wrong = sum(1 for r in ans if r["top_rerank_score"] < t)
            print(f"  {t:>8.3f}              | {refused:>2}/{len(unans)} ({100 * refused / len(unans):>3.0f}%)"
                  f"       | {wrong:>2}/{len(ans)} ({100 * wrong / len(ans):>3.0f}%)")

    out = Path(f"eval/results_{args.label}.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved per-question results to {out}")


if __name__ == "__main__":
    main()