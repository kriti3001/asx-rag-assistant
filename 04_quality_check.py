"""
Quality check: print the full extracted text for a specific page so you can
eyeball whether headers are clean and tables read sensibly.

Usage: python 04_quality_check.py extracted/cba_extracted.json 115
(if no page number given, shows a few pages with tables spread across the doc)
"""
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python 04_quality_check.py extracted/cba_extracted.json [page_number]")
    sys.exit(1)

with open(sys.argv[1], encoding="utf-8") as f:
    docs = json.load(f)

by_page = {d["metadata"]["page"]: d for d in docs}

if len(sys.argv) >= 3:
    page_num = sys.argv[2]  # page labels can be plain numbers or "54a"/"54b" for split spread pages
    if page_num in by_page:
        d = by_page[page_num]
        print(f"=== Page {page_num} | num_tables={d['metadata']['num_tables']} ===\n")
        print(d["text"])
    else:
        print(f"Page {page_num} was not extracted (likely filtered as near-empty).")
else:
    # Show 3 pages with tables, spread across the document, for a quick scan
    table_pages = [d for d in docs if d["metadata"]["num_tables"] > 0]
    if not table_pages:
        print("No pages with tables found at all -- something is wrong.")
        sys.exit(0)

    sample = [table_pages[0], table_pages[len(table_pages)//2], table_pages[-1]]
    for d in sample:
        print(f"\n{'='*60}")
        print(f"Page {d['metadata']['page']} | num_tables={d['metadata']['num_tables']}")
        print(f"{'='*60}")
        print(d["text"][:600])