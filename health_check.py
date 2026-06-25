"""
Full pipeline health check -- verifies every layer is correctly set up
and working, end to end. Run this any time you want a single clear
pass/fail picture instead of piecing it together from scattered tests.

Usage: python health_check.py
"""

import os
import sys
import json
import glob

CHECKS_PASSED = []
CHECKS_FAILED = []


def check(label, condition, detail=""):
    if condition:
        CHECKS_PASSED.append(label)
        print(f"  [OK]   {label}")
    else:
        CHECKS_FAILED.append((label, detail))
        print(f"  [FAIL] {label}{' -- ' + detail if detail else ''}")


def main():
    print("=" * 60)
    print("ASX RAG ASSISTANT -- FULL HEALTH CHECK")
    print("=" * 60)

    # --- 1. Environment ---
    print("\n[1] Environment")
    check(".env file exists", os.path.exists(".env"))
    load_dotenv_ok = False
    try:
        from dotenv import load_dotenv
        load_dotenv()
        load_dotenv_ok = True
    except ImportError:
        pass
    check("python-dotenv installed", load_dotenv_ok)
    check("GROQ_API_KEY is set", bool(os.getenv("GROQ_API_KEY")))

    for pkg in ["pdfplumber", "langchain_text_splitters", "sentence_transformers", "chromadb", "groq", "streamlit"]:
        try:
            __import__(pkg)
            check(f"package '{pkg}' importable", True)
        except ImportError:
            check(f"package '{pkg}' importable", False, "run: pip install " + pkg.replace("_", "-"))

    # --- 2. Source data ---
    print("\n[2] Source PDFs")
    expected_pdfs = ["cba", "bhp", "csl", "woolworths", "telstra"]
    found_pdfs = glob.glob("data/*.pdf")
    for company in expected_pdfs:
        matched = any(company in os.path.basename(f).lower() for f in found_pdfs)
        check(f"data/*{company}*.pdf present", matched)

    # --- 3. Extraction ---
    print("\n[3] Extraction output (extracted/)")
    extracted_files = glob.glob("extracted/*_extracted.json")
    check("extracted/ contains files", len(extracted_files) >= 5, f"found {len(extracted_files)}")
    total_extracted_pages = 0
    for f in extracted_files:
        try:
            with open(f, encoding="utf-8") as fh:
                docs = json.load(fh)
            total_extracted_pages += len(docs)
            check(f"{os.path.basename(f)} loads and has content", len(docs) > 0)
        except Exception as e:
            check(f"{os.path.basename(f)} loads and has content", False, str(e))
    print(f"  Total extracted pages across all files: {total_extracted_pages}")

    # --- 4. Chunking ---
    print("\n[4] Chunking output (chunks/)")
    chunk_files = glob.glob("chunks/*_chunks.json")
    check("chunks/ contains files", len(chunk_files) >= 5, f"found {len(chunk_files)}")
    total_chunks = 0
    has_embedding_text_field = True
    for f in chunk_files:
        try:
            with open(f, encoding="utf-8") as fh:
                chunks = json.load(fh)
            total_chunks += len(chunks)
            check(f"{os.path.basename(f)} loads and has content", len(chunks) > 0)
            if chunks and "embedding_text" not in chunks[0]:
                has_embedding_text_field = False
        except Exception as e:
            check(f"{os.path.basename(f)} loads and has content", False, str(e))
    check("chunks include embedding_text field (company-aware embeddings)", has_embedding_text_field)
    print(f"  Total chunks across all files: {total_chunks}")

    # --- 5. ChromaDB ---
    print("\n[5] ChromaDB vector store")
    chroma_ok = False
    collection_count = 0
    collection = None
    try:
        import chromadb
        client = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_collection(name="asx_annual_reports")
        collection_count = collection.count()
        chroma_ok = collection_count > 0
    except Exception as e:
        check("ChromaDB collection exists and has data", False, str(e))
    else:
        check("ChromaDB collection exists and has data", chroma_ok, f"{collection_count} items")
        check(
            "ChromaDB item count roughly matches chunk count",
            abs(collection_count - total_chunks) < 50 if total_chunks else False,
            f"chunks={total_chunks}, chroma={collection_count}",
        )

    # --- 6. Embedding model ---
    print("\n[6] Embedding model")
    embed_ok = False
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        test_vec = model.encode(["test sentence"])
        embed_ok = test_vec.shape[1] == 384
    except Exception as e:
        check("Embedding model loads and produces 384-dim vectors", False, str(e))
    else:
        check("Embedding model loads and produces 384-dim vectors", embed_ok)

    # --- 7. Retrieval ---
    print("\n[7] Retrieval (live test query)")
    if chroma_ok and embed_ok and collection is not None and model is not None:
        try:
            query_vec = model.encode(["What was Woolworths employee benefits expense"]).tolist()
            results = collection.query(query_embeddings=query_vec, n_results=3)
            retrieval_ok = len(results["ids"][0]) > 0
        except Exception as e:
            check("Retrieval returns results for a real query", False, str(e))
        else:
            check("Retrieval returns results for a real query", retrieval_ok)
    else:
        check("Retrieval returns results for a real query", False, "skipped -- ChromaDB or embedding model not ready")

    # --- 8. Groq connectivity ---
    print("\n[8] Groq API connectivity")
    if os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # use the cheap fallback model just for this connectivity check
                messages=[{"role": "user", "content": "Reply with the single word: OK"}],
                temperature=0,
            )
            groq_ok = bool(response.choices[0].message.content)
        except Exception as e:
            check("Groq API call succeeds", False, str(e))
        else:
            check("Groq API call succeeds", groq_ok)
    else:
        check("Groq API call succeeds", False, "skipped -- no API key")

    # --- 9. Application files ---
    print("\n[9] Application files present")
    for f in ["09_rag_chatbot.py", "10_streamlit_app.py", "eval/eval_cases.py", "eval/run_eval.py"]:
        check(f"{f} exists", os.path.exists(f))

    # --- Summary ---
    print("\n" + "=" * 60)
    total = len(CHECKS_PASSED) + len(CHECKS_FAILED)
    print(f"SUMMARY: {len(CHECKS_PASSED)}/{total} checks passed")
    print("=" * 60)
    if CHECKS_FAILED:
        print("\nFailed checks:")
        for label, detail in CHECKS_FAILED:
            print(f"  - {label}{': ' + detail if detail else ''}")
        sys.exit(1)
    else:
        print("\nAll checks passed -- the full pipeline is working end to end.")


if __name__ == "__main__":
    main()
