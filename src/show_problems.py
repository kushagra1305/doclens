"""
src/show_problems.py - list the questions an end-to-end run got wrong.

Usage (from the doclens/ folder):
    python src/show_problems.py qwen7b_full

Shows, for answerable questions that were refused or cited the wrong page, and
for unanswerable questions that were answered: the question, what was expected,
what the system cited, and the start of its answer.
"""
import json
import sys
from pathlib import Path

label = sys.argv[1] if len(sys.argv) > 1 else "qwen7b_full"
rows = json.loads(Path(f"eval/e2e_{label}.json").read_text(encoding="utf-8"))

problems = [
    r for r in rows
    if (r["answerable"] and (r["refused_by"] or r["citation_correct"] is False))
    or (not r["answerable"] and not r["refused_by"])
]
print(f"{len(problems)} problem questions in run '{label}'\n")
for r in problems:
    kind = f"REFUSED by {r['refused_by']}" if r["refused_by"] else "WRONG PAGE CITED"
    if not r["answerable"]:
        kind = "ANSWERED but should refuse"
    print(f"{r['id']} [{kind}] reranker score {r['top_score']}")
    print(f"  Q: {r['question']}")
    if r["answerable"]:
        short = r["expected"].replace("Income-tax-Act-2025_2026_2026-08-25_10-41-45_a1d3cd_en.pdf", "Act")
        print(f"  expected: {short}")
        cited = [f"p{s['page']}" for s in r["sources"]]
        print(f"  cited   : {', '.join(cited) if cited else '-'}")
    print(f"  answer  : {r['answer'][:220].replace(chr(10), ' ')}")
    print()