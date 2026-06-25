"""
Step 6 (Phase 5 final): Full RAG pipeline -- retrieval + answer generation.

Takes a question, retrieves relevant chunks from ChromaDB (with company
filtering when the question names a company -- see detect_company), builds
a prompt with that context, and sends it to Groq's free LLM API to
generate a real natural-language answer with citations back to the
source company and page.

Usage:
  python 09_rag_chatbot.py "What was CBA's net profit in FY25?"
  python 09_rag_chatbot.py "What was Woolworths employee benefits expense?" --k 5
"""

import os
import re
import sys
import time

import chromadb
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "asx_annual_reports"
GROQ_MODEL = "llama-3.3-70b-versatile"
# Used automatically when GROQ_MODEL hits its rate limit (confirmed during
# development: the 70B model's free-tier DAILY token cap -- 100,000
# tokens/day -- is easy to exhaust during normal iterative testing, not
# just heavy production use). This smaller model has a much higher daily
# allowance, so falling back to it keeps the assistant usable instead of
# hard-failing for the rest of the day.
FALLBACK_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_K = 5

# Groq's free tier caps llama-3.3-70b-versatile at 12,000 tokens PER MINUTE
# (a hard limit -- confirmed by hitting a 413 error mid-development with a
# ~16,800 token comparison-query prompt). We cap our own prompt size well
# under that, rather than hoping a fixed retrieval k happens to fit --
# chunk sizes vary a lot (a single large table chunk can be 4,000+ chars),
# so a fixed chunk COUNT doesn't reliably bound token count the way a
# direct character budget does. ~6000 tokens (~24000 chars) leaves
# comfortable headroom for instructions, formatting, the question itself,
# and the model's own output, while still allowing several real chunks
# per company in comparison mode.
MAX_CONTEXT_CHARS = 24000

# The fallback model has an even STRICTER per-minute limit than the
# primary one -- confirmed by a real 413 error during testing: 6,000 TPM
# vs the primary model's 12,000 TPM. A prompt that fits the primary
# model's budget can still be too large once we've already fallen back,
# so we shrink the context further specifically for fallback calls.
FALLBACK_MAX_CONTEXT_CHARS = 10000

# A distance above this is unlikely to be genuinely relevant (based on our
# own testing: strong matches cluster 0.4-0.7, weak/borderline matches
# push toward 0.8+). Chunks beyond this are still shown to the LLM as
# context, but the LLM is instructed to treat low-relevance context with
# appropriate skepticism rather than forcing an answer from it.
RELEVANCE_DISTANCE_CUTOFF = 0.85

COMPANY_ALIASES = {
    "CBA": ["cba", "commonwealth bank", "commbank"],
    "BHP": ["bhp"],
    "CSL": ["csl"],
    "WOOLWORTHS": ["woolworths", "woolies", "wow"],
    "TELSTRA": ["telstra", "tls"],
}


def detect_company(query):
    """Same logic as 08_test_retrieval.py -- see that file for rationale."""
    query_lower = query.lower()
    matched = set()
    for company, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            if alias in query_lower:
                matched.add(company)
                break
    if len(matched) == 1:
        return matched.pop()
    return None


_COMPARISON_WORDS = [
    "compare", "comparison", "across the companies", "across companies",
    "each company", "all companies", "all five", "versus", " vs ", " vs.",
    "between the companies", "which company",
]


# Different companies label the same financial concept differently --
# confirmed directly from the real reports: BHP uses "Profit after
# taxation" and never says "net profit" as a labeled figure anywhere;
# CSL explicitly uses "Net Profit After Tax"; Woolworths uses "Profit for
# the period". A query asking for "net profit" embeds closer to whichever
# company's wording happens to be closest, which silently crowds out
# companies using different (but equivalent) terminology -- this is
# exactly what we saw happen to BHP in testing. We mitigate this by
# expanding the QUERY used for embedding (not the displayed query) to
# include common synonyms, giving every company's phrasing a fair chance
# to match regardless of which specific term the user typed.
FINANCIAL_TERM_SYNONYMS = {
    "net profit": ["profit after tax", "profit after taxation", "NPAT", "profit for the period", "profit for the year"],
    "revenue": ["total revenue", "total income", "sales"],
    "dividend": ["dividend per share", "distribution"],
}


def expand_query_for_embedding(query):
    """
    Append known synonyms for any financial term found in the query, to
    use specifically as the EMBEDDING input (not what's shown to the
    user or the LLM as the literal question). This is a generic fix that
    applies across all companies' reports, not a company-specific patch.
    """
    query_lower = query.lower()
    additions = []
    for term, synonyms in FINANCIAL_TERM_SYNONYMS.items():
        if term in query_lower:
            additions.extend(synonyms)

    if not additions:
        return query

    return query + " (" + ", ".join(additions) + ")"


def looks_like_comparison(query):
    """
    Detect queries that want results spread across multiple/all companies
    (e.g. "Compare net profit across the companies") rather than a single
    company's figure. This matters because a single generic embedding
    search across all 3,219 chunks lets companies with more naturally
    matching phrasing crowd out the others -- we confirmed this directly:
    a comparison query returned mostly Telstra/CBA chunks while Woolworths
    barely made the list and BHP didn't appear at all, even though every
    company's report has the relevant figure. Detecting this case lets us
    retrieve per-company instead (see retrieve_multi_company), guaranteeing
    each company gets a fair chance to contribute its own best-matching
    chunk rather than competing in one global ranking.
    """
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in _COMPARISON_WORDS)


def retrieve(model, collection, query, k=DEFAULT_K):
    embedding_query = expand_query_for_embedding(query)
    query_embedding = model.encode([embedding_query]).tolist()
    company_filter = detect_company(query)
    where = {"company": company_filter} if company_filter else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=where,
    )

    chunks = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        chunks.append({
            "chunk_id": chunk_id,
            "text": doc,
            "company": meta["company"],
            "page": meta["page"],
            "source_file": meta["source_file"],
            "distance": dist,
        })

    return chunks


def retrieve_multi_company(model, collection, query, per_company_k=2):
    """
    Run a SEPARATE retrieval for each known company and combine the
    results, instead of one global search across all companies' chunks.
    This guarantees every company gets a fair chance to surface its own
    best-matching chunk(s) for the query, rather than competing in one
    ranking where a company with more naturally-matching phrasing
    (e.g. "Profit for the year attributable to" scoring closer to a
    generic "net profit" query than another company's different wording
    for the same concept) crowds out the others entirely.
    """
    query_embedding = model.encode([expand_query_for_embedding(query)]).tolist()
    all_chunks = []

    for company in COMPANY_ALIASES:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=per_company_k,
            where={"company": company},
        )
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": doc,
                "company": meta["company"],
                "page": meta["page"],
                "source_file": meta["source_file"],
                "distance": dist,
            })

    # Sort by company for a clean, grouped presentation (not by distance --
    # we deliberately want one section per company, not a single ranking
    # that could still bury a weaker-but-correct company match at the bottom).
    all_chunks.sort(key=lambda c: c["company"])
    return all_chunks


def enforce_context_budget(chunks, max_chars=MAX_CONTEXT_CHARS):
    """
    Trim the chunk list so total context text stays under max_chars,
    preventing a 413 "request too large" error from Groq's free-tier TPM
    limit (confirmed during testing: a 5-companies x 5-chunks comparison
    prompt hit ~16,800 tokens against a 12,000 TPM cap). Rather than
    relying on a fixed chunk COUNT to bound size -- chunk sizes vary a lot,
    a single large table chunk can be 4,000+ characters -- we measure
    actual text length and drop chunks once the budget is used up.

    Prioritization: process company-by-company in round-robin order
    (taking each company's best/first chunk before any company's second
    chunk, and so on), so if trimming is needed, every represented
    company keeps at least one chunk rather than one company's chunks
    filling the whole budget before others get a turn. This matters most
    for comparison queries, where balanced coverage is the whole point.
    """
    by_company = {}
    for c in chunks:
        by_company.setdefault(c["company"], []).append(c)

    # Within each company, keep original (distance) order -- best match first.
    max_len = max((len(v) for v in by_company.values()), default=0)
    round_robin = []
    for round_idx in range(max_len):
        for company_chunks in by_company.values():
            if round_idx < len(company_chunks):
                round_robin.append(company_chunks[round_idx])

    kept = []
    total_chars = 0
    for c in round_robin:
        chunk_len = len(c["text"])
        if total_chars + chunk_len > max_chars and kept:
            # Keep at least one chunk overall even if it alone exceeds
            # budget (better than returning nothing), but otherwise stop
            # once adding the next chunk would exceed the cap.
            continue
        kept.append(c)
        total_chars += chunk_len

    return kept


def build_prompt(query, chunks, max_chars=MAX_CONTEXT_CHARS):
    """
    Build the prompt sent to the LLM: instructions + retrieved context +
    the question. Each chunk is labeled with its source (company, page)
    so the LLM can cite specific sources in its answer, and so we can
    verify those citations are real afterward.

    max_chars is configurable (not a hardcoded constant) so the SAME
    chunk list can be re-packed into a smaller prompt for the fallback
    model, which has a stricter per-minute token limit than the primary
    model -- confirmed by a real 413 error during testing where a prompt
    that fit the primary model's budget still exceeded the fallback's.
    """
    chunks = enforce_context_budget(chunks, max_chars=max_chars)

    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        relevance_note = "" if c["distance"] <= RELEVANCE_DISTANCE_CUTOFF else " (low relevance match)"
        context_blocks.append(
            f"[Source {i}: {c['company']}, page {c['page']}{relevance_note}]\n{c['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a financial research assistant answering questions about ASX-listed companies' annual reports, using only the context provided below.

IMPORTANT DATA QUALITY WARNING: Some sources contain TWO versions of the same table -- a clean version (plain sentences, numbers in order, e.g. "Profit after taxation attributable to BHP shareholders 9,019 7,897 12,921") and a CORRUPTED version of the SAME data further down, broken up with " | " characters (e.g. "Profit after taxa | tion attributable to B | HP shareh | olde | rs | 12,921"). The corrupted version is a PDF extraction artifact: its numbers are frequently MISSING, OUT OF ORDER, or attached to the wrong row. Whenever you see " | " characters inside a sentence, that sentence is corrupted -- find the clean version of the same fact elsewhere in the same source and use that instead. NEVER cite a number that appears inside a " | "-broken line if a clean version of the same row exists anywhere in the context.

Context from annual reports:
{context}

Question: {query}

Instructions:
- Answer using ONLY the information in the context above. Do not use outside knowledge.
- Before answering, scan each relevant source for whether the figure you're about to use appears in a clean sentence or a "|"-broken one. If "|"-broken, search the rest of that same source for the clean version of that row first.
- If the context does not contain enough information to answer confidently, say so clearly rather than guessing.
- Prefer a DIRECTLY STATED figure over calculating one yourself. If a source states "net profit was $X million" or "profit for the period was $X million" directly, use that number as-is -- do not add, subtract, or combine it with other line items unless no direct figure exists at all.
- When comparing the same kind of figure ACROSS MULTIPLE companies, use the same consistent metric for every company (e.g. always the company's TOTAL profit figure, not one company's total and another's shareholder-attributable-only portion) -- inconsistent metric choice makes the comparison meaningless even if each individual number is technically correct.
- When you state a figure or fact, cite which source it came from using the format [Source N]. Before writing [Source N], re-check that source's text and confirm the exact number you are about to state actually appears in it -- do not cite a source out of habit (e.g. always citing Source 1) or because it ranked first; cite whichever specific source's text genuinely contains the number.
- Be concise and direct. Lead with the answer, then add supporting detail if useful.
- If sources marked "(low relevance match)" don't actually help answer the question, ignore them rather than forcing an answer from them.
"""
    return prompt, chunks


def generate_answer(client, query, chunks):
    """
    Try the primary model first with the normally-sized prompt. On a
    rate-limit error (confirmed to happen on Groq's free tier from normal
    iterative testing, not just heavy load -- see FALLBACK_GROQ_MODEL's
    comment), retry with the higher-quota fallback model -- but REBUILD
    the prompt with a smaller budget first, since the fallback model has
    an even stricter per-minute limit than the primary one (confirmed by
    a real 413 error: a prompt sized for the primary model's 12,000 TPM
    still exceeded the fallback's 6,000 TPM). Returns (answer_text,
    model_used, chunks_used) so callers can be transparent about which
    model and prompt actually produced the answer, and can display
    sources that match exactly what the model saw (the fallback path
    uses a smaller, re-trimmed chunk list than the primary path).
    """
    prompt, chunks_used = build_prompt(query, chunks, max_chars=MAX_CONTEXT_CHARS)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # low temperature: we want grounded, consistent answers, not creative ones
        )
        return response.choices[0].message.content, GROQ_MODEL, chunks_used
    except RateLimitError:
        fallback_prompt, fallback_chunks_used = build_prompt(query, chunks, max_chars=FALLBACK_MAX_CONTEXT_CHARS)
        # The fallback model has its own (separate, much smaller) rate
        # limit and can itself be rate-limited -- confirmed by a real
        # crash during testing (uncaught RateLimitError on the fallback
        # call). Its per-minute limit resets quickly (the actual error
        # we hit suggested under 1 second), so a short retry loop handles
        # this transient case instead of crashing outright.
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=FALLBACK_GROQ_MODEL,
                    messages=[{"role": "user", "content": fallback_prompt}],
                    temperature=0.1,
                )
                return response.choices[0].message.content, FALLBACK_GROQ_MODEL, fallback_chunks_used
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff


_INLINE_CITATION_PATTERN = re.compile(r"([^.]*?)\[?Source\s+(\d+)[^\]]*\]?\.?", re.IGNORECASE)
_NUMBER_NEAR_CITATION_PATTERN = re.compile(r"[\d,]+\.?\d*")


def verify_citations(answer, chunks_used):
    """
    Lightweight post-generation check: for each '[Source N]' citation in
    the answer, look at the text immediately preceding it for a number,
    and check whether that number actually appears in chunk N's text.

    This is a real, confirmed failure mode (not hypothetical): in testing,
    a model cited "[Source 1]" for a genuine $2,343 million figure, but
    Source 1 was actually just the report's cover page ("Telstra / Annual
    Report") -- the number was real, but the citation was wrong, likely
    because Source 1 happened to rank closest by embedding distance even
    though its content was irrelevant. Prompt instructions alone don't
    reliably prevent this (confirmed: the instruction was already in
    place when this happened), so this catches it after the fact instead
    of trusting every citation blindly.

    Returns a list of warning strings (empty if no issues found) rather
    than silently failing -- the caller decides how to surface this
    (e.g. an inline caveat in the UI).
    """
    warnings = []
    for sentence_before, source_num_str in _INLINE_CITATION_PATTERN.findall(answer):
        numbers_in_sentence = _NUMBER_NEAR_CITATION_PATTERN.findall(sentence_before)
        # Only check sentences that actually claim a specific figure --
        # citations on purely qualitative statements have nothing
        # numeric to verify, so skip those rather than false-flag them.
        numbers_in_sentence = [n for n in numbers_in_sentence if any(c.isdigit() for c in n)]
        if not numbers_in_sentence:
            continue

        source_idx = int(source_num_str) - 1
        if source_idx < 0 or source_idx >= len(chunks_used):
            warnings.append(
                f"Citation [Source {source_num_str}] does not correspond to any retrieved source."
            )
            continue

        source_text = chunks_used[source_idx]["text"]
        source_numbers = set(_NUMBER_NEAR_CITATION_PATTERN.findall(source_text))
        if not any(n in source_numbers for n in numbers_in_sentence):
            warnings.append(
                f"Citation [Source {source_num_str}] may not support the figure(s) "
                f"{numbers_in_sentence} mentioned near it -- that source "
                f"({chunks_used[source_idx]['company']} page {chunks_used[source_idx]['page']}) "
                f"does not appear to contain a matching number."
            )
    return warnings


def get_answer(model, collection, groq_client, query, k=DEFAULT_K):
    """
    Core RAG logic: retrieve, build prompt, generate answer. Returns a
    plain dict (not printed) so both the CLI (main(), below) and the
    Streamlit app (10_streamlit_app.py) can reuse this exact same code
    path -- one source of truth for retrieval + generation, rather than
    duplicating this logic for each interface.
    """
    if looks_like_comparison(query):
        # Comparison queries are genuinely harder: each company phrases
        # "net profit" differently, and for some companies the figure
        # lives in a messy, fragmented summary-table chunk that doesn't
        # embed as cleanly as a clean narrative sentence would. A higher
        # per-company k gives each company more chances to surface its
        # relevant chunk even if it's not the single closest match.
        chunks = retrieve_multi_company(model, collection, query, per_company_k=5)
        company_filter = None
        mode_note = "(comparison mode: retrieved separately per company)"
    else:
        chunks = retrieve(model, collection, query, k=k)
        company_filter = detect_company(query)
        mode_note = f"(restricted to company: {company_filter})" if company_filter else None

    if not chunks:
        return {
            "answer": "No relevant chunks found at all (filter may be too restrictive).",
            "mode_note": mode_note,
            "sources": [],
            "model_used": None,
        }

    answer, model_used, chunks_used = generate_answer(groq_client, query, chunks)

    citation_warnings = verify_citations(answer, chunks_used)

    fallback_note = (
        f"(Note: primary model was rate-limited; answered using fallback model {FALLBACK_GROQ_MODEL})"
        if model_used == FALLBACK_GROQ_MODEL else None
    )

    return {
        "answer": answer,
        "mode_note": mode_note,
        "fallback_note": fallback_note,
        "citation_warnings": citation_warnings,
        "model_used": model_used,
        "sources": [
            {
                "label": f"Source {i}",
                "company": c["company"],
                "page": c["page"],
                "chunk_id": c["chunk_id"],
                "distance": c["distance"],
                "low_relevance": c["distance"] > RELEVANCE_DISTANCE_CUTOFF,
                "text": c["text"],
            }
            for i, c in enumerate(chunks_used, start=1)
        ],
    }


def answer_question(model, collection, groq_client, query, k=DEFAULT_K, show_sources=True):
    """CLI wrapper: calls get_answer() and prints the result to the console."""
    result = get_answer(model, collection, groq_client, query, k=k)

    print(f"\n{'='*70}")
    print(f"QUESTION: {query}")
    if result["mode_note"]:
        print(result["mode_note"])
    if result.get("fallback_note"):
        print(result["fallback_note"])
    print(f"{'='*70}\n")
    print(result["answer"])

    if show_sources and result["sources"]:
        print(f"\n{'-'*70}")
        print("Sources retrieved:")
        for s in result["sources"]:
            flag = " [low relevance]" if s["low_relevance"] else ""
            print(f"  [{s['label']}] {s['company']} page {s['page']} ({s['chunk_id']}, distance={s['distance']:.3f}){flag}")

    if result.get("citation_warnings"):
        print(f"\n{'-'*70}")
        print("⚠️  Citation warnings:")
        for w in result["citation_warnings"]:
            print(f"  - {w}")


def main():
    args = sys.argv[1:]
    k = DEFAULT_K
    if "--k" in args:
        idx = args.index("--k")
        k = int(args[idx + 1])
        del args[idx:idx + 2]

    query = " ".join(args) if args else None
    if not query:
        print("Usage: python 09_rag_chatbot.py \"your question here\" [--k N]")
        sys.exit(1)

    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure your .env file has "
            "GROQ_API_KEY=your_key_here in this folder."
        )
    groq_client = Groq(api_key=groq_api_key)

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    answer_question(model, collection, groq_client, query, k=k)


if __name__ == "__main__":
    main()