"""
Inspect the full answer text and citation detail for a specific eval case.

Usage: python eval/inspect_result.py woolworths_profit
"""
import json
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python eval/inspect_result.py <case_id>")
    sys.exit(1)

case_id = sys.argv[1]
results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")

with open(results_path, encoding="utf-8") as f:
    data = json.load(f)

for r in data["results"]:
    if r["id"] == case_id:
        print(f"Question: {r['question']}")
        print(f"\nAnswer:\n{r['answer']}")
        print(f"\nMatched (answer-text check): {r['matched']}")
        print(f"Citation check passed: {r['citation_check_passed']}")
        print(f"Citation check detail: {r['citation_check_detail']}")
        print(f"\nSources retrieved:")
        for s in r["sources"]:
            print(f"  {s['company']} page {s['page']} (distance={s['distance']})")
        break
else:
    print(f"No result found with id '{case_id}'")
