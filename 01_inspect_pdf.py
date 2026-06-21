"""
Step 1: Inspect a PDF before extracting anything.
Run this on each of your 5 PDFs to understand what you're working with
(page count, whether text extraction works cleanly, how many tables exist).
"""

import sys
import pdfplumber

def inspect_pdf(filepath):
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")

    with pdfplumber.open(filepath) as pdf:
        num_pages = len(pdf.pages)
        print(f"Total pages: {num_pages}")

        # Sample a few pages spread across the document
        sample_pages = [0, num_pages // 4, num_pages // 2, (3 * num_pages) // 4, num_pages - 1]
        sample_pages = sorted(set(p for p in sample_pages if 0 <= p < num_pages))

        for page_num in sample_pages:
            page = pdf.pages[page_num]
            text = page.extract_text() or ""
            tables = page.extract_tables()

            print(f"\n--- Page {page_num + 1} ---")
            print(f"Text length: {len(text)} characters")
            print(f"Tables found: {len(tables)}")
            print(f"First 200 chars of text:\n{text[:200]!r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 01_inspect_pdf.py data/your_file.pdf")
        sys.exit(1)

    inspect_pdf(sys.argv[1])
