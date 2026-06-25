"""
Diagnostic: inspect the RAW extracted text for a specific page, BEFORE
chunking touches it. Use this to isolate whether a problem originates in
extraction (02_extract_pdf.py) or chunking (05_chunk_documents.py).

Usage: python 11_inspect_extracted_page.py extracted/bhp_extracted.json 84
"""
import json
import sys

if len(sys.argv) < 3:
    print("Usage: python 11_inspect_extracted_page.py extracted/company_extracted.json page_label")
    sys.exit(1)

filepath = sys.argv[1]
page_label = sys.argv[2]

with open(filepath, encoding="utf-8") as f:
    docs = json.load(f)

matches = [d for d in docs if d["metadata"]["page"] == page_label]

print(f"Found {len(matches)} page document(s) for page {page_label}\n")
for d in matches:
    print(f"=== metadata: {d['metadata']} ===")
    print(d["text"])
    print()
