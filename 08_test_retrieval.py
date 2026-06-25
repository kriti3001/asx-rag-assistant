"""
Step 5 (Phase 5): Test retrieval quality against the real ChromaDB collection.

This queries the vector store with realistic financial questions and shows
the top-k retrieved chunks, so you can eyeball whether retrieval is
actually finding the right content before we wire up the LLM for full
answer generation.

Usage:
  python 08_test_retrieval.py "What was CBA's net profit in FY25?"
  python 08_test_retrieval.py "What was BHP's revenue from copper?" --k 3
  python 08_test_retrieval.py  (runs a few built-in sample questions if no query given)
"""

import sys
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "asx_annual_reports"

# Maps each company's ChromaDB metadata value to the aliases/names a user
# might naturally type. Checked case-insensitively against the query.
# This lets "How much did Woolworths spend on..." automatically restrict
# the vector search to only WOOLWORTHS chunks, preventing cross-company
# bleed-through we saw during testing (e.g. a Woolworths question
# returning CBA/Telstra chunks just because they share similar wording).
COMPANY_ALIASES = {
    "CBA": ["cba", "commonwealth bank", "commbank"],
    "BHP": ["bhp"],
    "CSL": ["csl"],
    "WOOLWORTHS": ["woolworths", "woolies", "wow"],
    "TELSTRA": ["telstra", "tls"],
}


def detect_company(query):
    """
    Check whether the query mentions a known company by name or alias.
    Returns the canonical metadata company value (e.g. "CBA") if exactly
    one company is mentioned, or None if zero or multiple companies are
    detected -- in the multiple-company case (e.g. "compare X and Y"),
    we deliberately don't filter, since the user wants both.
    """
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


SAMPLE_QUESTIONS = [
    "What was CBA's net profit after tax?",
    "What was BHP's revenue from copper?",
    "What is Woolworths' dividend per share?",
    "What are CSL's main business segments?",
    "What was Telstra's total revenue?",
]


def run_query(model, collection, query, k=5):
    query_embedding = model.encode([query]).tolist()

    company_filter = detect_company(query)
    where = {"company": company_filter} if company_filter else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=where,
    )

    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    if company_filter:
        print(f"(filtered to company: {company_filter})")
    print(f"{'='*70}")

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not ids:
        print("\nNo results found (filter may be too restrictive, or company has no matching chunks).")
        return

    for rank, (chunk_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), start=1):
        # ChromaDB returns distance (lower = more similar) by default for
        # its default space; we show it as-is rather than converting, to
        # avoid implying a precision we don't have.
        print(f"\n--- Rank {rank} | {meta['company']} page {meta['page']} | distance={dist:.3f} ---")
        preview = doc[:250].replace("\n", " ")
        print(preview)


def main():
    args = sys.argv[1:]
    k = 5
    if "--k" in args:
        idx = args.index("--k")
        k = int(args[idx + 1])
        del args[idx:idx + 2]

    query = " ".join(args) if args else None

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection has {collection.count()} chunks.\n")

    if query:
        run_query(model, collection, query, k=k)
    else:
        print("No query given -- running built-in sample questions:\n")
        for q in SAMPLE_QUESTIONS:
            run_query(model, collection, q, k=3)


if __name__ == "__main__":
    main()