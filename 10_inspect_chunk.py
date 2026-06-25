"""
Diagnostic: inspect a specific page's chunk(s) to verify what content
the LLM actually saw, useful for debugging citation issues.

Usage: python 10_inspect_chunk.py chunks/bhp_chunks.json 84
"""
import json
import sys

if len(sys.argv) < 3:
    print("Usage: python 10_inspect_chunk.py chunks/company_chunks.json page_label")
    sys.exit(1)

filepath = sys.argv[1]
page_label = sys.argv[2]

with open(filepath, encoding="utf-8") as f:
    chunks = json.load(f)

matches = [c for c in chunks if c["metadata"]["page"] == page_label]

print(f"Found {len(matches)} chunk(s) on page {page_label}\n")
for c in matches:
    print(f"=== {c['chunk_id']} (is_table={c['is_table']}, {c['metadata']['char_count']} chars) ===")
    print(c["text"])
    print()
