"""
Step 2: Optimised PDF extraction for ASX annual reports.

Handles:
1. Header/footer stripping (repeated boilerplate lines across pages)
2. Tables converted to clean Markdown (not raw list-of-lists)
3. Tables kept attached to their surrounding narrative text, in page order
4. Empty/near-empty pages skipped
5. Metadata tagging (company, source file, page number) on every chunk

Output: a list of "page documents" — one dict per page, each containing
clean combined text (narrative + markdown tables) and metadata.
This is the input the chunking step (next file) will consume.
"""

import os
import re
import json
from collections import Counter
import pdfplumber


def extract_company_name(filename):
    """Turn 'cba_annual_report_2025.pdf' into 'CBA' for metadata."""
    base = os.path.basename(filename)
    name = base.split("_")[0]
    return name.upper()


_DOUBLED_CHAR_PATTERN = re.compile(r"([A-Za-z])\1{1,}")


def has_doubled_chars(text, min_hits=4):
    """
    Cheap detector for a character-rendering defect seen in some PDFs
    (e.g. Telstra's), where letters repeat back-to-back: "NNootteess ttoo"
    or "TTTeeelll" instead of "Notes to" / "Tel". Normal English text
    occasionally has 1-2 incidental double letters (e.g. "Total", "Press"),
    but 4+ in one string is essentially never genuine text -- validated
    against real sentences from these reports, which topped out at 2.
    This is just a regex scan (cheap), so it's safe to run on every page;
    the expensive dedupe_chars() fix is only applied where this is positive.
    """
    if not text:
        return False
    return len(_DOUBLED_CHAR_PATTERN.findall(text)) >= min_hits


def extract_page_text(page):
    """
    Extract text from a page. We deliberately do NOT call dedupe_chars()
    here, even on pages with the doubled-character rendering defect
    (Telstra's report): we confirmed that defect only ever affects the
    running header line, never genuine content, and dedupe_chars() is
    expensive enough (roughly 2-3x slower per page) that calling it on
    every affected page made full-document extraction unacceptably slow.
    Instead, clean_page_text()'s first-line looks_garbled() check strips
    the corrupted header line cheaply, without touching real content.
    """
    return page.extract_text() or ""


def find_repeated_lines(page_units, sample_size=60, min_repeat_ratio=0.15, edge_window=5):
    """
    Detect boilerplate header/footer lines by sampling pages and finding
    short-ish lines that repeat across a meaningful fraction of them.
    e.g. 'Notes to the Financial Statements' or 'For the year ended 30 June 2025'
    showing up on dozens of pages is noise, not content.

    A wider edge_window (5 instead of 3) and lower threshold (0.15 instead of
    0.4) are needed because long reports often have several DIFFERENT
    section headers (e.g. "Notes to the Financial Statements" in one part,
    "Statements of Changes in Equity" in another) — no single line repeats
    on a huge fraction of ALL pages, only within its own section.

    Takes page_units (the post-split list of (page, label) tuples) rather
    than raw pdf.pages, so double-wide spread pages are sampled correctly
    as their split halves, not as one interleaved wide page.
    """
    line_counter = Counter()
    total_units = len(page_units)
    sample_indices = range(0, total_units, max(1, total_units // sample_size))

    sampled = 0
    for i in sample_indices:
        page, _label = page_units[i]
        text = extract_page_text(page)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        edge_lines = lines[:edge_window] + lines[-edge_window:]
        for line in edge_lines:
            # Skip pure-number lines here -- handled separately by regex in
            # clean_page_text. Skip very long lines -- real headers/footers
            # are short; long repeated lines are coincidental body text.
            if 3 < len(line) <= 90:
                line_counter[line] += 1
        sampled += 1

    threshold = max(2, int(sampled * min_repeat_ratio))
    repeated = {line for line, count in line_counter.items() if count >= threshold}
    return repeated


def looks_garbled(line, min_hits=4):
    """
    Looser version of has_doubled_chars for cleaning a single line: some
    PDFs (e.g. Telstra's) have a character-rendering defect confined to
    their running header line, where the exact garbling pattern varies
    slightly page to page (e.g. 'SNotets toa thet financ...' on one page,
    'INontes cto thoe finam...' on another). This means exact-match
    repeated-line detection can't catch it. Since we've confirmed this
    defect only affects the header line and never genuine content, we use
    a lower threshold here (1 hit instead of 3) specifically for
    identifying and dropping a garbled FIRST line of a page.
    """
    return has_doubled_chars(line, min_hits=min_hits)


def clean_page_text(text, repeated_lines):
    """Remove boilerplate lines and strip standalone page-number lines."""
    if not text:
        return ""

    lines = text.split("\n")
    cleaned = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in repeated_lines:
            continue
        # standalone page numbers (just digits, possibly short)
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        # The first line of a page is the most common place for a running
        # header to appear. If it looks garbled (character-rendering
        # defect), drop it -- we've confirmed this defect never affects
        # genuine content, only this kind of boilerplate header line.
        if idx == 0 and looks_garbled(stripped):
            continue
        cleaned.append(stripped)

    return "\n".join(cleaned)


def page_has_real_table(page, min_filled_ratio=0.3, min_rows=2):
    """
    Detect whether a page genuinely contains a table, using pdfplumber's
    DEFAULT (lines/lines) strategy, which requires actual ruled lines to
    form a table. This is a much more reliable detector than the 'text'
    strategy: infographic-style pages (stat boxes, pull-quotes, magazine
    layouts) rarely have real ruled grid lines, so they correctly produce
    no tables or empty/sparse ones here -- even though the 'text' strategy
    (needed later for good label/value parsing) falsely fragments these
    same pages into garbage "tables".

    We only treat a page as having a real table if the default strategy
    finds at least one table with enough rows and a reasonable fraction
    of non-empty cells.
    """
    default_tables = page.extract_tables()  # default = lines/lines
    for t in default_tables:
        if len(t) < min_rows:
            continue
        total = sum(len(row) for row in t)
        filled = sum(1 for row in t for c in row if c and c.strip())
        if total > 0 and (filled / total) >= min_filled_ratio:
            return True
    return False


def table_to_text(table):
    """
    Convert a pdfplumber table into readable text that preserves the
    label-to-value relationship per row, WITHOUT forcing a rigid Markdown
    grid. This matters because financial-statement tables in real ASX
    reports often have merged/irregular cells (a label cell spanning
    several lines, paired with a value cell containing several stacked
    numbers separated by \\n). Forcing these into a strict grid loses the
    pairing; instead we split each multi-line cell on '\\n' and zip the
    pieces back together as "label value" lines, which keeps the
    information an LLM needs even though it's not a pretty visual table.
    """
    if not table:
        return ""

    lines_out = []
    for row in table:
        cells = [(cell or "").strip() for cell in row]
        cells = [c for c in cells if c]
        if not cells:
            continue

        cell_line_lists = [c.split("\n") for c in cells]
        max_lines = max(len(cl) for cl in cell_line_lists)

        if max_lines == 1:
            lines_out.append(" | ".join(cells))
        else:
            for i in range(max_lines):
                parts = []
                for cl in cell_line_lists:
                    if i < len(cl) and cl[i].strip():
                        parts.append(cl[i].strip())
                if parts:
                    lines_out.append(" | ".join(parts))

    return "\n".join(lines_out)


def get_page_units(pdf, wide_page_ratio=1.3):
    """
    Some ASX reports (we've seen this with Telstra's) store each physical
    page as a two-page spread merged into one wide PDF page object --
    e.g. "Comprehensive Income" and "Financial Position" side by side at
    double width. Extracting text from these directly interleaves both
    pages' content into nonsense.

    Detect unusually wide pages (width much greater than height, beyond
    a normal landscape ratio) and split them into left/right halves,
    each treated as its own page for extraction purposes. Normal
    portrait/landscape pages are returned unchanged.

    Returns a list of (page_object, page_label) tuples, where page_label
    is the printed page number, or e.g. "54a"/"54b" for a split spread.
    """
    units = []
    for page_num, page in enumerate(pdf.pages, start=1):
        is_wide_spread = page.width > page.height * wide_page_ratio
        if is_wide_spread:
            half_width = page.width / 2
            left = page.crop((0, 0, half_width, page.height))
            right = page.crop((half_width, 0, page.width, page.height))
            units.append((left, f"{page_num}a"))
            units.append((right, f"{page_num}b"))
        else:
            units.append((page, str(page_num)))
    return units


def extract_pdf(filepath, min_chars=20):
    """
    Extract one PDF into a list of page-level documents.
    Each document = {text, metadata} where text combines cleaned narrative
    text and any tables (as Markdown), and metadata identifies source.
    """
    company = extract_company_name(filepath)
    page_docs = []

    with pdfplumber.open(filepath) as pdf:
        total_pages = len(pdf.pages)
        page_units = get_page_units(pdf)
        repeated_lines = find_repeated_lines(page_units)

        for page, page_label in page_units:
            # Some PDFs have an overlapping-text-layer bug that doubles
            # every character; extract_page_text() detects and fixes this
            # cheaply, only paying the expensive dedupe cost when needed.
            raw_text = extract_page_text(page)
            cleaned_text = clean_page_text(raw_text, repeated_lines)

            table_blocks = []
            table_cell_texts = set()

            if page_has_real_table(page):
                # Only re-parse with the text/lines strategy (better at
                # pairing multi-line labels with their values) once we've
                # confirmed via the default strategy that a real table
                # genuinely exists on this page.
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "lines",
                })
                for t in tables:
                    txt = table_to_text(t)
                    if txt:
                        table_blocks.append(txt)
                        for row in t:
                            for cell in row:
                                if cell:
                                    for piece in cell.strip().split("\n"):
                                        if piece.strip():
                                            table_cell_texts.add(piece.strip())

            # pdfplumber's extract_text() picks up table cell content as
            # plain text too, which would duplicate it alongside the Markdown
            # version below. Strip lines that are just table cell content
            # (short lines exactly matching a known cell) to avoid that.
            if table_cell_texts:
                lines = cleaned_text.split("\n")
                lines = [l for l in lines if l.strip() not in table_cell_texts]
                cleaned_text = "\n".join(lines)

            # Combine narrative text and tables for this page, in order.
            # Keeping them together (not in separate chunks) preserves the
            # link between a sentence like "Net profit increased to:" and
            # the table that follows it.
            combined_parts = []
            if cleaned_text:
                combined_parts.append(cleaned_text)
            combined_parts.extend(table_blocks)

            combined_text = "\n\n".join(combined_parts)

            # Skip pages with effectively no content (e.g. blank cover/back pages)
            if len(combined_text) < min_chars:
                continue

            page_docs.append({
                "text": combined_text,
                "metadata": {
                    "company": company,
                    "source_file": os.path.basename(filepath),
                    "page": page_label,
                    "total_pages": total_pages,
                    "num_tables": len(table_blocks),
                }
            })

    return page_docs


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 02_extract_pdf.py data/your_file.pdf")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Extracting: {filepath}")

    docs = extract_pdf(filepath)

    print(f"\nExtracted {len(docs)} non-empty pages (out of total pages in PDF)")
    print(f"Pages with at least one table: {sum(1 for d in docs if d['metadata']['num_tables'] > 0)}")

    # Show one example so you can sanity-check quality
    if docs:
        example = docs[len(docs) // 2]  # a middle page, likely to have real content
        print(f"\n--- Example: page {example['metadata']['page']} ---")
        print(f"Metadata: {example['metadata']}")
        print(f"Text preview (first 500 chars):\n{example['text'][:500]}")

    # Save full extraction to disk as JSON for the next step (chunking) to use
    out_dir = "extracted"
    os.makedirs(out_dir, exist_ok=True)
    company = extract_company_name(filepath)
    out_path = os.path.join(out_dir, f"{company.lower()}_extracted.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)
    print(f"\nSaved extraction to: {out_path}")
