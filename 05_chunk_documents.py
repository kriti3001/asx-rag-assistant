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

# Self-contained copy of the doubled-character corruption detector from
# 02_extract_pdf.py (used there to strip garbled headers, used here to
# avoid splitting already-garbled segments -- see split_merged_subsections).
# Kept as a local copy rather than cross-importing a numbered filename,
# which would be a fragile, unusual dependency between scripts.
_DOUBLED_CHAR_PATTERN = re.compile(r"([A-Za-z])\1{1,}")


def is_garbled_text(segment, max_rate=0.08, min_len=20):
    """
    Rate-based corruption check: counts doubled-letter occurrences PER
    CHARACTER, not as a raw count. A raw count threshold works fine for
    short single lines (where we know roughly how long a header line is),
    but doesn't generalize to longer multi-line segments -- a long,
    completely clean financial table can easily accumulate several
    incidental double letters across words like "Statement", "Total",
    "Loss", purely from its length, and a fixed count would wrongly flag
    it as corrupted. Measured on real examples from these reports:
    genuinely clean multi-line text scores well under 1% (0.008), while
    truly garbled text (Telstra's rendering defect) scores 25%+ (0.26).
    A max_rate of 0.08 gives a comfortable margin between the two.
    """
    if not segment or len(segment) < min_len:
        return False
    matches = len(_DOUBLED_CHAR_PATTERN.findall(segment))
    return (matches / len(segment)) >= max_rate

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


_SUBSECTION_HEADER_PATTERN = re.compile(r"^\d+\.\d+\.\d+\s+[A-Z]", re.MULTILINE)


def split_merged_subsections(segment, min_sections=2):
    """
    Some table segments span multiple distinct numbered sub-notes that
    happen to sit on the same page with no blank line between them (e.g.
    "2.3.1 Branch and administration expenses", "2.3.2 Employee benefits
    expense", "2.3.3 Depreciation and amortisation expense" all running
    consecutively). Treating these as one unsplittable table chunk dilutes
    each sub-note's content with the other two, hurting retrieval
    precision for a query about just one of them (e.g. "employee benefits
    expense" competing for relevance against two unrelated expense notes
    in the same chunk).

    If a segment contains 2+ lines matching a "N.N.N Title" sub-section
    header pattern, split it into separate pieces at those boundaries so
    each sub-note becomes its own (still unsplit-internally) chunk. If
    there's only 0-1 such header (the normal case -- a chunk legitimately
    starting with its own single section header), leave it as one segment.

    Guard: some PDFs (e.g. Telstra's table-of-contents pages) have a
    character-rendering defect that doubles letters throughout the WHOLE
    segment, not just a header line. A garbled segment can coincidentally
    contain digit patterns that look like "N.N.N" section numbers (e.g.
    from mangled page-reference numbers in a contents listing). Splitting
    already-garbled text just produces more small garbled fragments
    instead of one larger one -- no information is gained, so we skip
    splitting entirely when the segment looks corrupted.
    """
    if is_garbled_text(segment):
        return [segment]

    matches = list(_SUBSECTION_HEADER_PATTERN.finditer(segment))
    if len(matches) < min_sections:
        return [segment]

    pieces = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(segment)
        piece = segment[start:end].strip()
        if piece:
            pieces.append(piece)

    # Anything before the first detected header (rare, but possible if the
    # segment has introductory text before the first numbered sub-note)
    # gets prepended to the first piece rather than dropped.
    leading = segment[:matches[0].start()].strip()
    if leading and pieces:
        pieces[0] = leading + "\n" + pieces[0]

    if not pieces:
        return [segment]

    # Whole-segment garbling can be diluted below threshold when a large
    # mixed segment has both clean and corrupted sections averaged
    # together (confirmed on a real Telstra page). Check each individual
    # piece too -- splitting can isolate a smaller, more concentrated
    # corrupted fragment that the whole-segment average missed. If any
    # piece looks garbled, the split isn't trustworthy here, so fall back
    # to the single whole segment rather than emit a mix of good and
    # garbled small chunks.
    if any(is_garbled_text(p, min_len=10) for p in pieces):
        return [segment]

    return pieces


def split_into_segments(page_text):
    """
    Split a page's combined text into a list of (segment_text, is_table)
    tuples. Segments are separated by blank lines, which is how
    02_extract_pdf.py joins narrative text and table blocks together.
    Table-like segments are additionally checked for merged sub-sections
    (see split_merged_subsections) and broken apart if needed.
    """
    raw_segments = re.split(r"\n\s*\n", page_text)
    segments = []
    for seg in raw_segments:
        seg = seg.strip()
        if not seg:
            continue
        if is_table_segment(seg):
            for piece in split_merged_subsections(seg):
                segments.append((piece, True))
        else:
            segments.append((seg, False))
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
        company = doc["metadata"]["company"]
        for pc in page_chunks:
            chunk_counter += 1
            # Many individual financial-statement notes never mention the
            # company name in their own text (e.g. a note titled "Employee
            # benefits expense" just lists figures, with "Woolworths" only
            # appearing on page headers elsewhere that got stripped during
            # extraction). Since only embedded TEXT is compared for
            # similarity -- metadata like company is not used by the
            # embedding model -- a query that includes the company name
            # (a very common, natural way to ask) can fail to retrieve the
            # right chunk purely because the company name isn't present in
            # it to match against. We fix this generally for every chunk,
            # in every company's report, by prepending the company name to
            # a SEPARATE embedding_text field used only for generating the
            # embedding vector. The original "text" field (shown to users,
            # used for citations and for the LLM's context) stays exactly
            # as extracted, so this doesn't change what the person sees.
            embedding_text = f"{company}: {pc['text']}"

            all_chunks.append({
                "chunk_id": f"{company.lower()}_{chunk_counter:05d}",
                "text": pc["text"],
                "embedding_text": embedding_text,
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
