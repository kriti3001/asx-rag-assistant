"""
Step 2: Optimised PDF extraction for ASX annual reports.

Handles:
1. Header/footer stripping (repeated boilerplate lines across pages,
   plus a fallback for garbled first-line headers caused by a character-
   rendering defect seen in some PDFs)
2. Tables converted to readable "label value" text (preserving row
   structure without forcing a rigid grid, since real financial tables
   often have merged/irregular multi-line cells)
3. Tables kept attached to their surrounding narrative text, in page order
4. Empty/near-empty pages skipped
5. Double-wide "two-page-spread" PDFs (seen in some reports) detected and
   split into separate left/right page units before extraction
6. Metadata tagging (company, source file, page number, table count) on
   every page document

Output: a list of "page documents" — one dict per page (or per half-page,
for split spreads), each containing clean combined text (narrative text +
table text) and metadata. This is the input the chunking step
(05_chunk_documents.py) consumes.
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


def is_plausible_page_number(stripped_line, min_year=1990, max_year=2035):
    """
    A standalone short digit string could be a genuine page number (safe
    to strip) OR a fiscal year used as a table column header on its own
    line (must NOT be stripped). Confirmed directly on BHP's report: its
    tables show "2025" / "2024" / "2023" each on a separate line before
    the data rows, and our previous page-number filter was deleting all
    of them -- removing year context entirely, not just leaving it
    ambiguous. A page number in any of these ~100-450 page reports will
    never realistically fall in a plausible fiscal year range, so
    excluding that range is a safe way to keep stripping real page
    numbers while preserving year labels.
    """
    if not re.fullmatch(r"\d{1,4}", stripped_line):
        return False
    value = int(stripped_line)
    if min_year <= value <= max_year:
        return False  # looks like a year, not a page number
    return True


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
        # standalone page numbers (just digits, possibly short) -- but
        # NOT standalone years, which must be preserved as table header
        # context (see is_plausible_page_number).
        if is_plausible_page_number(stripped):
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


_YEAR_LINE_PATTERN = re.compile(r"^((?:20\d{2}\s*)+)$", re.MULTILINE)
_NUMBER_PATTERN = re.compile(r"-?[\d,]+\.?\d*")


def annotate_year_columns(table_text, max_year_span=3):
    """
    Financial-statement tables show fiscal years as column headers, but in
    several DIFFERENT formats across these reports -- confirmed directly:
    Woolworths uses two years on one line ("2025 2024"); BHP uses three
    years on one line ("2025 2024 2023") in some tables and the same
    three years stacked one-per-line elsewhere. An LLM reading a data row
    with several numbers (e.g. "Profit after taxation 11,143 9,601
    14,324") has no reliable way to know which number is which year
    without this header context -- confirmed in testing, where this
    caused a wrong-year citation (the LLM picked the prior year's figure
    instead of the current year's for BHP specifically).

    Rather than match one fixed year-count per pattern (which we tried
    twice and both times a real report used a different count/layout),
    this matches ANY line consisting purely of one or more 4-digit years
    in sequence -- 2, 3, or more -- and annotates it generically. A
    max_year_span sanity check (the highest and lowest year on the line
    must be within a few years of each other) avoids misfiring on
    unrelated multi-year content like a sustainability target table
    listing "2025 2030", which is not a fiscal-year comparison header.
    """
    def replace_line(match):
        years = match.group(1).split()
        if len(years) < 2:
            return match.group(0)
        if max(int(y) for y in years) - min(int(y) for y in years) > max_year_span:
            return match.group(0)
        ordinal_note = ", ".join(
            f"number {i+1} is for {y}" for i, y in enumerate(years)
        )
        return match.group(0) + f"\n[Note: in the data rows below, {ordinal_note}]"

    return _YEAR_LINE_PATTERN.sub(replace_line, table_text)


_CANONICAL_PROFIT_PATTERN = re.compile(
    r"^(Profit(?:/\(loss\))? for the (?:period|year)\b)(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_ATTRIBUTABLE_PATTERN = re.compile(
    r"^(Profit(?:/\(loss\))? (?:for the (?:period|year) )?attributable to\b|"
    r"^Net profit attributable to\b|"
    r"^Equity holders of\b)",
    re.IGNORECASE,
)


def annotate_canonical_profit_line(table_text):
    """
    Statutory profit/loss statements consistently show a Group/consolidated
    TOTAL line (e.g. "Profit for the period 953 117") immediately followed
    by a BREAKDOWN of that same total split between equity holders and
    non-controlling interests (e.g. "Profit/(loss) for the period
    attributable to: Equity holders of the parent entity 963 108").

    Both numbers are genuinely correct, correctly labeled figures for the
    same period -- this isn't a data error. But a query like "what was
    the company's profit" is ambiguous between the two, and we confirmed
    in testing that an LLM can inconsistently pick either one (run to run,
    even with explicit prompt instructions to be consistent) -- which
    silently breaks comparability the moment one company's answer uses
    the total and another's uses the shareholder-attributable figure.

    Rather than rely on probabilistic prompt-level guidance, we resolve
    this deterministically at the data layer: detect the canonical
    "Profit for the period/year" line and tag it explicitly as the
    standard headline figure to report by default, distinguishing it
    from the immediately-following "...attributable to..." breakdown.
    This makes the answer the SAME every time, for every company, since
    it's now a rule rather than a per-query LLM judgment call.
    """
    lines = table_text.split("\n")
    out = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        match = _CANONICAL_PROFIT_PATTERN.match(stripped)
        # Guard: a genuine table row has a real number after the label
        # (e.g. "953 117"). Ordinary prose that happens to start with
        # similar wording (e.g. "Profit for the year ahead looks strong
        # according to guidance") does not, and must NOT be annotated --
        # confirmed as a real false-positive via direct testing.
        has_number_after_label = bool(match) and bool(_NUMBER_PATTERN.search(match.group(2) or ""))
        if match and has_number_after_label and not _ATTRIBUTABLE_PATTERN.search(stripped):
            out.append(line)
            out.append(
                "[Note: the line above is the company's TOTAL/Group profit figure -- "
                "this is the standard headline 'net profit' figure to report by default. "
                "Lines below mentioning 'attributable to equity holders' or 'non-controlling "
                "interests' are a BREAKDOWN of this same total, not a different total -- "
                "do not report the attributable-only portion as if it were the company's "
                "overall net profit unless specifically asked for the shareholder-attributable figure.]"
            )
        else:
            out.append(line)
    return "\n".join(out)


def _extract_numbers(text):
    """Pull out numeric tokens (handles thousands separators, decimals, negatives)."""
    return [n for n in _NUMBER_PATTERN.findall(text) if any(c.isdigit() for c in n)]


def _line_duplicates_table_row(line, table_row_number_sets, min_overlap=2):
    """
    Check whether a plain-text line is substantially the same content as
    a table row already captured in table_blocks, even if exact-string
    matching would miss it. Confirmed as a real bug: pdfplumber's
    'text' table-parsing strategy fragments a row like "Profit after
    taxation 11,143 9,601 14,324" into pieces like "Profit after
    taxatio" | "n" | "9,601" | "14,324", so individual fragments never
    exactly match the plain-text line, and the dedup silently failed --
    leaving the SAME financial figures duplicated in both a clean table
    chunk and a messy leftover narrative-text chunk. The duplicate
    chunk lacks our year-column annotation and is missing context, so
    when it gets retrieved instead of the clean version, an LLM has no
    way to know which number is which year -- confirmed as the direct
    cause of a wrong-year citation in testing.

    Numbers are a much more robust signal here than exact text: a real
    duplicate row shares most of its numeric values with the
    corresponding table row regardless of how the surrounding label text
    got fragmented.
    """
    line_numbers = set(_extract_numbers(line))
    if len(line_numbers) < min_overlap:
        return False
    for row_numbers in table_row_number_sets:
        overlap = line_numbers & row_numbers
        if len(overlap) >= min_overlap:
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
    text and any tables (as text), and metadata identifies source.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")

    company = extract_company_name(filepath)
    page_docs = []

    try:
        pdf_context = pdfplumber.open(filepath)
    except Exception as e:
        raise RuntimeError(
            f"Could not open '{filepath}' as a PDF. It may be corrupted, "
            f"password-protected, or not a real PDF file. Original error: {e}"
        )

    with pdf_context as pdf:
        total_pages = len(pdf.pages)
        page_units = get_page_units(pdf)
        repeated_lines = find_repeated_lines(page_units)

        for page, page_label in page_units:
            # extract_page_text() returns plain extract_text() output; the
            # garbled-header defect some PDFs show (Telstra's) is handled
            # separately below via clean_page_text()'s first-line check,
            # not by re-extracting with dedupe_chars (too slow at scale --
            # see has_doubled_chars()'s docstring for why).
            raw_text = extract_page_text(page)
            cleaned_text = clean_page_text(raw_text, repeated_lines)

            cleaned_lines = cleaned_text.split("\n")
            cleaned_line_numbers = [set(_extract_numbers(l)) for l in cleaned_lines]

            table_blocks = []

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
                    if not txt:
                        continue

                    # For each row in this table-parsed version, check
                    # whether the plain-text extraction already covers it
                    # AT LEAST as completely. We confirmed a real case
                    # (BHP page 84) where the table-parsing strategy
                    # actually produces a WORSE result than plain text --
                    # a misdetected column boundary orphaned an entire
                    # year's figures into the header row, leaving the
                    # "official" table row with fewer numbers than the
                    # plain-text line covering the same content. Keeping
                    # BOTH versions as separate chunks doesn't fix this --
                    # it just lets retrieval gamble on which one it picks,
                    # and we confirmed the broken one can still win that
                    # gamble. So instead: if plain text covers this table
                    # at least as well, suppress the table_blocks version
                    # entirely rather than emit a worse duplicate chunk.
                    row_number_sets = []
                    for row in t:
                        row_text = " ".join(cell or "" for cell in row)
                        row_numbers = set(_extract_numbers(row_text))
                        if row_numbers:
                            row_number_sets.append(row_numbers)

                    rows_covered_by_plain_text = 0
                    for row_numbers in row_number_sets:
                        for line_numbers in cleaned_line_numbers:
                            overlap = row_numbers & line_numbers
                            if len(overlap) >= 2 and len(line_numbers) >= len(row_numbers):
                                rows_covered_by_plain_text += 1
                                break

                    # If the plain text already covers most of this
                    # table's real (numeric) rows at least as completely,
                    # the table-parsed version adds no value and risks
                    # being the worse copy -- skip it.
                    if row_number_sets and rows_covered_by_plain_text >= len(row_number_sets) * 0.6:
                        continue

                    table_blocks.append(txt)

            cleaned_text = "\n".join(cleaned_lines)

            # Combine narrative text and tables for this page, in order.
            # Keeping them together (not in separate chunks) preserves the
            # link between a sentence like "Net profit increased to:" and
            # the table that follows it.
            combined_parts = []
            if cleaned_text:
                combined_parts.append(cleaned_text)
            combined_parts.extend(table_blocks)

            combined_text = "\n\n".join(combined_parts)
            combined_text = annotate_year_columns(combined_text)
            combined_text = annotate_canonical_profit_line(combined_text)

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