"""
src/evaluate_e2e.py - Step 8 of DocLens: end-to-end test of the full pipeline.

Runs every question in eval/questions.json through generate.answer() (retrieval,
reranker gate, language model) and reports:
  - answerable questions: how many were answered, how often the cited page is
    the right one, and how many were wrongly refused (by the gate or the model)
  - unanswerable questions: how many were correctly refused

It cannot judge whether the wording of an answer is correct (a model can cite
the right page and still misread it), so all answers are saved to
eval/e2e_<label>.json for you to read and check by hand.

Usage (from the doclens/ folder):
    python src/evaluate_e2e.py --label llama3b
    python src/evaluate_e2e.py --label qwen7b --model qwen2.5:7b
    python src/evaluate_e2e.py --label quick --step 3     # every 3rd question only

This calls the language model once per question, so it is slow (a few seconds
to a minute per question). Leave it running.
"""
import argparse
import json
import time
from pathlib import Path

import generate

QUESTIONS_FILE = Path("eval/questions.json")


def cited_page_is_right(result: dict, q: dict) -> bool:
    return any(s["source"] == q["source"] and s["page"] in q["pages"] for s in result["sources"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--model", default=None, help="Ollama model name (default: the one in generate.py)")
    parser.add_argument("--step", type=int, default=1, help="use every Nth question (1 = all)")
    args = parser.parse_args()

    if args.model:
        generate.MODEL = args.model
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))[:: args.step]
    print(f"Model: {generate.MODEL} | cutoff: {generate.RERANK_CUTOFF} | {len(questions)} questions\n")

    rows = []
    for i, q in enumerate(questions, start=1):
        start = time.time()
        res = generate.answer(q["question"])
        secs = time.time() - start
        refused = res["refused_by"] is not None
        row = {
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "expected": f"{q['source']} p{q['pages']}" if q["answerable"] else None,
            "refused_by": res["refused_by"],
            "cited": res["cited"],
            "citation_correct": cited_page_is_right(res, q) if q["answerable"] and not refused else None,
            "top_score": round(res["top_score"], 2) if res["top_score"] is not None else None,
            "answer": res["answer"],
            "sources": res["sources"],
            "seconds": round(secs, 1),
        }
        rows.append(row)
        status = f"refused by {res['refused_by']}" if refused else ("answered, cite " + ("OK" if row["citation_correct"] else "WRONG/none")
                                                                     if q["answerable"] else "ANSWERED (should refuse)")
        print(f"[{i}/{len(questions)}] {q['id']} {status} ({secs:.0f}s)")

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if r["refused_by"] is None]
    correct = [r for r in answered if r["citation_correct"]]

    print("\n=== Answerable questions ===")
    print(f"  answered               : {len(answered)}/{len(ans)}")
    print(f"  cited the right page   : {len(correct)}/{len(answered)} of those answered")
    print(f"  wrongly refused by gate : {sum(1 for r in ans if r['refused_by'] == 'score_gate')}")
    print(f"  wrongly refused by model: {sum(1 for r in ans if r['refused_by'] == 'model')}")
    print("\n=== Unanswerable questions ===")
    print(f"  correctly refused      : {sum(1 for r in unans if r['refused_by'])}/{len(unans)}"
          f"  (gate {sum(1 for r in unans if r['refused_by'] == 'score_gate')},"
          f" model {sum(1 for r in unans if r['refused_by'] == 'model')})")
    for r in unans:
        if not r["refused_by"]:
            print(f"  ANSWERED when it should refuse: {r['question']}")

    out = Path(f"eval/e2e_{args.label}.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved all answers to {out} - open it and read a few answers against the cited pages.")


if __name__ == "__main__":
    main()