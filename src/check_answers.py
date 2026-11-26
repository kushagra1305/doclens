"""
src/check_answers.py - help you check answers by hand.

Usage (from the doclens/ folder):
    python src/check_answers.py qwen7b_full q01 q06 q13 q17 q31

For each question it prints the system's answer, then the full text of every
page it cited (taken from data/processed/pages.jsonl), so you can compare the
two side by side. Look for anything flipped, wrong or invented.
"""
import json
import sys
import textwrap
from pathlib import Path

if len(sys.argv) < 3:
    raise SystemExit("Usage: python src/check_answers.py <label> <id> [<id> ...]")

label, wanted = sys.argv[1], set(sys.argv[2:])
rows = json.loads(Path(f"eval/e2e_{label}.json").read_text(encoding="utf-8"))

pages = {}
with Path("data/processed/pages.jsonl").open(encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        pages[(p["source"], p["page"])] = p["text"]

for r in rows:
    if r["id"] not in wanted:
        continue
    print("=" * 78)
    print(f"{r['id']}: {r['question']}\n")
    print("SYSTEM ANSWER:")
    print(textwrap.fill(r["answer"], 78, initial_indent="  ", subsequent_indent="  "))
    if r["refused_by"]:
        print(f"\n(refused by {r['refused_by']}; nothing to compare)\n")
        continue
    for s in r["sources"]:
        short = s["source"][:30]
        print(f"\n--- CITED: {short}, page {s['page']} (passages {s['passages']}) ---")
        text = pages.get((s["source"], s["page"]), "(page text not found)")
        print(textwrap.fill(text, 78))
    print()