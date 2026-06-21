"""
Step 3: Chunk extracted page documents into embedding-ready pieces.

Strategy:
1. Each page's text may contain narrative prose AND one or more table
   blocks (tables were combined into the page text during extraction,
   separated by blank lines -- see 02_extract_pdf.py).
2. We split each page into its "segments" (alternating prose and table
   blocks) by detecting table-like segments (many short lines, numeric-
   heavy) vs prose segments.
3. Table segments are NEVER split internally -- a table either fits
   whole into a chunk, or gets its own dedicated chunk (even if that
   makes the chunk larger than the target size). Splitting a table mid-
   row would destroy the label-to-value pairing we worked hard to extract.
4. Prose segments are split using LangChain's RecursiveCharacterTextSplitter
   (paragraph -> sentence -> word fallback) with overlap, so a sentence
   cut at a boundary still has context in the neighboring chunk.
5. Every resulting chunk carries the original page's metadata, plus a
   chunk_id and an is_table flag.

Usage: python 05_chunk_documents.py extracted/woolworths_extracted.json
Output: chunks/woolworths_chunks.json
"""

import os
import re
import json
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter

TARGET_CHUNK_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 100
# Rough chars-per-token approximation for English text (~4 chars/token);
# good enough for sizing chunks without pulling in a full tokenizer model.
CHARS_PER_TOKEN = 4
TARGET_CHUNK_CHARS = TARGET_CHUNK_TOKENS * CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN


def is_table_segment(segment, min_lines=2, numeric_line_ratio=0.4, max_fragment_ratio=0.3):
    """
    Heuristic: a segment "looks like a table" if it has several lines and
    a meaningful fraction of them contain digits (dollar values, note
    references, percentages) -- consistent with the "label value value"
    style rows real tables produce (whether from table_to_text() or from
    plain extract_text() picking up a simply-laid-out table).

    We additionally reject segments that look like FRAGMENTED text rather
    than real tabular data: lines containing " | " where the pieces around
    the pipe are short, broken word fragments (e.g. "1 | General informati
    | on") rather than a real label/value pair. This pattern shows up when
    a misdetected pdfplumber table chops sentences into nonsense columns
    -- we want those treated as prose (and run through the recursive
    splitter) rather than locked as an unsplittable "table" chunk.
    """
    lines = [l for l in segment.split("\n") if l.strip()]
    if len(lines) < min_lines:
        return False

    numeric_lines = sum(1 for l in lines if re.search(r"\d", l))
    if (numeric_lines / len(lines)) < numeric_line_ratio:
        return False

    # Check for the "fragmented pipe" signature: a pipe-joined line where
    # a piece ends mid-word (no trailing space before next piece starts
    # with a lowercase letter) suggests broken text, not a real table row.
    pipe_lines = [l for l in lines if " | " in l]
    if pipe_lines:
        fragment_like = 0
        for l in pipe_lines:
            pieces = [p.strip() for p in l.split("|")]
            for i in range(len(pieces) - 1):
                a, b = pieces[i], pieces[i + 1]
                if a and b and re.search(r"[a-z]$", a) and re.match(r"^[a-z]", b):
                    # lowercase-to-lowercase boundary with no natural break
                    # is a strong signal of a word split across a phantom
                    # column, e.g. "informati" | "on"
                    fragment_like += 1
                    break
        if pipe_lines and (fragment_like / len(pipe_lines)) > max_fragment_ratio:
            return False

    return True


def split_into_segments(page_text):
    """
    Split a page's combined text into a list of (segment_text, is_table)
    tuples. Segments are separated by blank lines, which is how
    02_extract_pdf.py joins narrative text and table blocks together.
    """
    raw_segments = re.split(r"\n\s*\n", page_text)
    segments = []
    for seg in raw_segments:
        seg = seg.strip()
        if not seg:
            continue
        segments.append((seg, is_table_segment(seg)))
    return segments


def chunk_page_document(doc, splitter):
    """
    Turn one page document into one or more chunks.
    Table segments become their own chunk (whole, never split).
    Consecutive prose segments are concatenated then run through the
    recursive splitter together, so short paragraphs don't each become
    their own tiny chunk.
    """
    segments = split_into_segments(doc["text"])
    chunks = []

    prose_buffer = []

    def flush_prose_buffer():
        if not prose_buffer:
            return
        combined = "\n\n".join(prose_buffer)
        prose_buffer.clear()
        for piece in splitter.split_text(combined):
            if piece.strip():
                chunks.append({"text": piece, "is_table": False})

    for seg_text, is_table in segments:
        if is_table:
            # Flush any accumulated prose first, so ordering is preserved
            flush_prose_buffer()
            chunks.append({"text": seg_text, "is_table": True})
        else:
            prose_buffer.append(seg_text)

    flush_prose_buffer()
    return chunks


def chunk_extracted_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        page_docs = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TARGET_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    chunk_counter = 0

    for doc in page_docs:
        page_chunks = chunk_page_document(doc, splitter)
        for pc in page_chunks:
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": f"{doc['metadata']['company'].lower()}_{chunk_counter:05d}",
                "text": pc["text"],
                "is_table": pc["is_table"],
                "metadata": {
                    **doc["metadata"],
                    "char_count": len(pc["text"]),
                }
            })

    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 05_chunk_documents.py extracted/your_company_extracted.json")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Chunking: {filepath}")

    chunks = chunk_extracted_file(filepath)

    table_chunks = sum(1 for c in chunks if c["is_table"])
    prose_chunks = len(chunks) - table_chunks
    avg_chars = sum(c["metadata"]["char_count"] for c in chunks) / len(chunks) if chunks else 0

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"  Prose chunks: {prose_chunks}")
    print(f"  Table chunks: {table_chunks}")
    print(f"  Average chunk size: {avg_chars:.0f} characters (~{avg_chars/CHARS_PER_TOKEN:.0f} tokens)")

    # Show one example of each kind
    example_prose = next((c for c in chunks if not c["is_table"]), None)
    example_table = next((c for c in chunks if c["is_table"]), None)

    if example_prose:
        print(f"\n--- Example prose chunk ({example_prose['chunk_id']}, page {example_prose['metadata']['page']}) ---")
        print(example_prose["text"][:300])

    if example_table:
        print(f"\n--- Example table chunk ({example_table['chunk_id']}, page {example_table['metadata']['page']}) ---")
        print(example_table["text"][:300])

    # Save
    out_dir = "chunks"
    os.makedirs(out_dir, exist_ok=True)
    company = chunks[0]["metadata"]["company"].lower() if chunks else "unknown"
    out_path = os.path.join(out_dir, f"{company}_chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"\nSaved chunks to: {out_path}")
