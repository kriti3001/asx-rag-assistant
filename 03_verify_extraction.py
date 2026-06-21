"""
Quick verification script: check extraction stats for any company's
extracted JSON file. Usage: python 03_verify_extraction.py extracted/cba_extracted.json
"""
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python 03_verify_extraction.py extracted/cba_extracted.json")
    sys.exit(1)

with open(sys.argv[1], encoding="utf-8") as f:
    docs = json.load(f)

print("Total non-empty pages:", len(docs))
print("Pages with tables:", sum(1 for d in docs if d["metadata"]["num_tables"] > 0))
