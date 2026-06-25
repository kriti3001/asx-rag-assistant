# ASX Annual Report RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about five ASX-listed companies' FY2025 annual reports — Commonwealth Bank (CBA), BHP Group, CSL Limited, Woolworths Group, and Telstra — by retrieving relevant passages from the actual PDFs and generating grounded, cited answers with an LLM.

Built end-to-end: PDF extraction → chunking → embeddings → vector search → LLM generation → web UI, with a 10-case evaluation suite and a live citation-accuracy checker.

```
"What was Woolworths' employee benefits expense?"
→ "$11,439 million [Source 1: WOOLWORTHS, page 61a]"
```

## Why this project

Most "RAG chatbot" tutorials work on clean Wikipedia text and call it done. Real financial PDFs are not clean: multi-page spreads merged into single pages, tables fragmented across phantom columns, multi-year figures with no clear header, and duplicate renderings of the same data that disagree with each other. This project is the result of building against five real, messy, 90–450 page annual reports and fixing every failure that surfaced along the way — not just the happy path.

## Demo

| Question type | Example |
|---|---|
| Single-company lookup | "What was BHP's revenue from copper?" → $20,044M, broken into Group production ($18,023M) + third-party ($2,021M) |
| Cross-company comparison | "Compare net profit across the companies" → retrieves separately per company, avoiding one company's report crowding out another's |
| Out-of-scope (should refuse) | "What was BHP's stock price on 1 January 2026?" → correctly declines, since annual reports don't contain daily share prices |

## Architecture

```
PDF (pdfplumber)
  → extraction (02_extract_pdf.py)       cleans headers, repairs tables, fixes year columns
  → chunking (05_chunk_documents.py)     splits into retrieval-sized pieces, tags by company
  → embedding (07_embed_and_store.py)    all-MiniLM-L6-v2 → ChromaDB (cosine similarity)
  → retrieval (09_rag_chatbot.py)        company filtering, comparison mode, query expansion
  → generation (09_rag_chatbot.py)       Groq / Llama 3.3 70B, grounded prompt, citations
  → UI (10_streamlit_app.py)             chat interface with source transparency
```

**Stack:** Python · pdfplumber · LangChain text splitters · sentence-transformers · ChromaDB · Groq (Llama 3.3 70B + Llama 3.1 8B fallback) · Streamlit

All free-tier: local embeddings (no API cost), Groq's free LLM tier, ChromaDB running locally on disk.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com/keys).

Place the five annual report PDFs in `data/` (named `cba_*.pdf`, `bhp_*.pdf`, `csl_*.pdf`, `woolworths_*.pdf`, `telstra_*.pdf`), then run the pipeline once:

```bash
python 02_extract_pdf.py data/cba_annual_report_2025.pdf
python 02_extract_pdf.py data/bhp_annual_report_2025.pdf
python 02_extract_pdf.py data/csl_annual_report_2025.pdf
python 02_extract_pdf.py data/woolworths_annual_report_2025.pdf
python 02_extract_pdf.py data/telstra_annual_report_2025.pdf

python 05_chunk_documents.py extracted/cba_extracted.json
python 05_chunk_documents.py extracted/bhp_extracted.json
python 05_chunk_documents.py extracted/csl_extracted.json
python 05_chunk_documents.py extracted/woolworths_extracted.json
python 05_chunk_documents.py extracted/telstra_extracted.json

python 07_embed_and_store.py
```

Then either:

```bash
python 09_rag_chatbot.py "What was CBA's net profit after tax?"   # CLI
streamlit run 10_streamlit_app.py                                  # web UI
python health_check.py                                             # verify everything's wired up
```

## Evaluation

A 10-case eval set (`eval/eval_cases.py`) covers single-fact lookups, cross-company comparisons, and questions the system should refuse to answer. Every expected value was verified directly against the source PDFs, not assumed.

```bash
python eval/run_eval.py
```

Current result: **9/10 passing**, including a separate **citation-accuracy check** that verifies the specific source an answer cites actually contains the figure claimed — not just that the right number appears somewhere in the answer text.

| Category | Result |
|---|---|
| Single-fact lookups | 5/6 (1 known limitation, see below) |
| Cross-company comparisons | 2/2 |
| Refusal on out-of-scope questions | 2/2 |
| Citation accuracy (verified against source text) | 5/6 |

## The engineering story

This project's value is mostly in what got found and fixed, not just the final architecture. A sample of real, confirmed bugs from development:

- **Table extraction produced a worse result than plain text.** On one BHP table, pdfplumber's column detector merged an entire year's figures into the wrong cell. The "official" parsed table was actually less reliable than the plain-text rendering of the same page — extraction now compares both and keeps whichever is more complete, per table.
- **Multi-year tables had no year labels at all.** A page-number-stripping filter was deleting standalone "2025" / "2024" lines because they looked like page numbers. This silently removed the only context distinguishing which column was which year — fixed by excluding a plausible fiscal-year range from that filter, then explicitly annotating year order in the text the LLM reads.
- **The same fact, two different correct numbers.** Financial statements report both a company's total profit *and* the portion attributable to shareholders, on adjacent lines. Both are real, correctly labeled, and genuinely different — and an LLM asked for "net profit" picked inconsistently between them, run to run. Fixed deterministically at the data layer: extraction now tags which line is the canonical headline figure, rather than leaving it to a per-query LLM guess.
- **Retrieval picked the wrong company.** A query naming one company could retrieve a different company's chunk if its wording happened to embed slightly closer. Fixed by detecting the named company in the query and filtering retrieval to that company's chunks only, plus embedding each chunk with its company name prepended so the chunk's own embedding "knows" which company it's from.
- **A correct answer, wrongly cited.** The LLM sometimes stated the right number but attributed it to the wrong source (once, literally the report's cover page). A live `verify_citations()` check now cross-references every `[Source N]` citation against that source's actual text and surfaces a warning when they don't match — catching errors a glance at the answer wouldn't reveal.
- **Free-tier API limits, hit for real.** Comparison queries against five companies' worth of context blew past Groq's per-minute token cap, and a day of iterative testing exhausted the daily cap entirely. Both are now handled: a context-budget trimmer keeps every prompt under the per-minute limit (preserving at least one chunk per company), and a fallback model with its own retry logic keeps the assistant usable when the primary model's daily quota runs out.

## Known limitations

- **Smaller fallback model is less reliable on hard disambiguation.** When the primary model (Llama 3.3 70B) is rate-limited, the system falls back to Llama 3.1 8B, which has repeatedly shown weaker handling of the multi-year-column ambiguity than the primary model — occasionally citing a prior year's figure instead of the current year's.
- **Citation misattribution is detected, not fully prevented.** `verify_citations()` reliably flags when a cited source doesn't support the claimed figure, but the underlying tendency (citing a high-ranked-but-irrelevant source) isn't eliminated by prompting alone.
- **A handful of section-divider pages in Telstra's report** show cosmetic text duplication from a character-rendering defect in the source PDF, isolated to non-financial title pages.
- **Comparison queries are slower** (40+ seconds) than single-company lookups (1–2 seconds), since they retrieve separately per company and build a larger prompt.

## Project structure

```
02_extract_pdf.py         PDF → cleaned text + repaired tables (per page)
05_chunk_documents.py     page text → retrieval-sized chunks, company-tagged
07_embed_and_store.py     chunks → embeddings → ChromaDB
08_test_retrieval.py      retrieval-only debugging tool (no LLM call)
09_rag_chatbot.py         full RAG pipeline: retrieval + Groq generation + citation checking
10_streamlit_app.py       web chat interface
health_check.py           one-command verification of every pipeline layer
eval/
  eval_cases.py           10 verified question/expected-answer pairs
  run_eval.py             runs the eval set, scores answers and citations
  inspect_result.py       inspect one eval case's full answer + sources
```

## Possible extensions

- Re-rank retrieved chunks with a cross-encoder before generation
- Add a query-rewriting step for ambiguous financial terminology beyond the current synonym list
- Deploy to Streamlit Community Cloud for a shareable live link
- Extend the eval set with adversarial phrasing and multi-hop questions
