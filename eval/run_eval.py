"""
Run the evaluation set against the live RAG pipeline and report results.

Usage: python eval/run_eval.py
Output: console summary + eval/eval_results.json (detailed results for
        later inspection or for charting accuracy over time as the
        pipeline improves).
"""

import os
import sys
import json
import re
import time

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

import importlib.util
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_eval_dir = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location(
    "rag_chatbot", os.path.join(_repo_root, "09_rag_chatbot.py")
)
rag_chatbot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rag_chatbot)

_cases_spec = importlib.util.spec_from_file_location(
    "eval_cases", os.path.join(_eval_dir, "eval_cases.py")
)
_eval_cases_module = importlib.util.module_from_spec(_cases_spec)
_cases_spec.loader.exec_module(_eval_cases_module)
EVAL_CASES = _eval_cases_module.EVAL_CASES


def normalize(text):
    """Lowercase and strip common punctuation/formatting so substring
    matching isn't thrown off by things like '$11,439 million' vs
    '11439' vs '11,439m'."""
    return re.sub(r"[^\w\s.]", "", text.lower())


def check_single_fact_or_comparison(answer, expected_substrings):
    """
    Pass if at least one expected substring appears in the answer.
    Numbers are checked both with and without commas, since LLM output
    formatting varies (e.g. "11,439" vs "11439").
    """
    answer_norm = normalize(answer)
    hits = []
    for substr in expected_substrings:
        substr_norm = normalize(substr)
        if substr_norm in answer_norm:
            hits.append(substr)
    return len(hits) > 0, hits


def check_refusal(answer, refusal_phrases):
    """Pass if the answer contains language indicating it declined to
    answer, rather than confidently stating a fabricated fact."""
    answer_lower = answer.lower()
    hits = [p for p in refusal_phrases if p.lower() in answer_lower]
    return len(hits) > 0, hits


_SOURCE_BRACKET_PATTERN = re.compile(r"\[?Source\s+([\d,\s]+)", re.IGNORECASE)
_DIGIT_PATTERN = re.compile(r"\d+")


def extract_cited_source_numbers(answer):
    """
    Find every 'Source N' reference in the answer text and return the
    set of distinct source numbers cited, e.g. {1, 3, 6}.

    Handles multiple real formats the LLM actually produces -- confirmed
    necessary after two real false-negatives in testing: (1) the model
    wrote "[Source 1: WOOLWORTHS, page 55a]" (extra detail inside the
    brackets) instead of the plain "[Source 1]" we originally expected,
    and (2) the model cited in plain prose with NO brackets at all, e.g.
    "...in Source 1: WOOLWORTHS, page 61a." The leading "[" is now
    optional rather than required. Only digits immediately following
    "Source" (comma/space-separated for multi-citations like
    "[Source 1, 3]") are captured -- digits appearing later in the
    surrounding text (e.g. "page 55a" containing "55") are correctly NOT
    treated as source numbers.
    """
    numbers = set()
    for bracket_match in _SOURCE_BRACKET_PATTERN.finditer(answer):
        inner = bracket_match.group(1)
        for digit_match in _DIGIT_PATTERN.finditer(inner):
            numbers.add(int(digit_match.group()))
    return numbers


def check_citation_accuracy(answer, sources, expected_substrings):
    """
    Stricter check than check_single_fact_or_comparison: verifies that
    at least one of the SOURCES EXPLICITLY CITED in the answer text
    (via "[Source N]") actually contains one of the expected substrings
    in its own chunk text. This catches a subtler failure mode than
    "is the right number anywhere in the answer": an answer could state
    a correct number while citing a source that doesn't actually support
    it (e.g. citing the wrong company's chunk, or a chunk that happens to
    not contain that figure at all) -- which would still pass a plain
    substring check on the answer, but represents an ungrounded citation.

    Returns (passed, matched_substrings, details) where details explains
    which cited source(s) actually backed up the claim, for transparency.
    """
    cited_numbers = extract_cited_source_numbers(answer)
    if not cited_numbers:
        return False, [], "No [Source N] citations found in answer to verify."

    matched = []
    details = []
    for source_num in sorted(cited_numbers):
        idx = source_num - 1  # sources are 1-indexed in the answer text
        if idx < 0 or idx >= len(sources):
            details.append(f"Source {source_num}: cited but does not exist in retrieved sources (hallucinated citation number)")
            continue
        source_text_norm = normalize(sources[idx]["text"])
        for substr in expected_substrings:
            substr_norm = normalize(substr)
            if substr_norm in source_text_norm:
                matched.append(substr)
                details.append(f"Source {source_num} ({sources[idx]['company']} page {sources[idx]['page']}): contains '{substr}' -- citation verified")
                break
        else:
            details.append(f"Source {source_num} ({sources[idx]['company']} page {sources[idx]['page']}): does NOT contain any expected figure")

    return len(matched) > 0, matched, "; ".join(details)


def run_eval():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Check your .env file.")
    groq_client = Groq(api_key=groq_api_key)

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=rag_chatbot.CHROMA_DB_PATH)
    collection = client.get_collection(name=rag_chatbot.COLLECTION_NAME)

    results = []
    passed = 0

    for case in EVAL_CASES:
        print(f"\nRunning: {case['id']} ({case['category']})...")
        t0 = time.time()
        result = rag_chatbot.get_answer(model, collection, groq_client, case["question"])
        elapsed = time.time() - t0
        answer = result["answer"]

        if case["category"] == "refusal":
            ok, hits = check_refusal(answer, case["refusal_phrases"])
            citation_ok, citation_hits, citation_detail = None, None, None
        else:
            ok, hits = check_single_fact_or_comparison(answer, case["expected_substrings"])
            if case["category"] == "single_fact":
                if not ok:
                    # The answer didn't even contain a correct figure, so
                    # there's nothing valid to verify a citation FOR.
                    # Confirmed necessary after a real false-positive: an
                    # answer stating the WRONG year's figure (CBA $9,394M
                    # instead of $10,116M) still got citation_ok=True,
                    # because the correct number happened to ALSO appear
                    # elsewhere in the same cited source's text -- the
                    # check was verifying "is this number somewhere in the
                    # source" rather than "does the source support what
                    # the answer actually claimed". A wrong answer cannot
                    # have an accurate citation by definition.
                    citation_ok, citation_hits, citation_detail = False, [], (
                        "Skipped: answer-text check failed, so there is no correct "
                        "claim for a citation to accurately support."
                    )
                else:
                    citation_ok, citation_hits, citation_detail = check_citation_accuracy(
                        answer, result["sources"], case["expected_substrings"]
                    )
            else:
                citation_ok, citation_hits, citation_detail = None, None, None

        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] ({elapsed:.1f}s) matched: {hits if hits else 'none'}")
        if citation_ok is not None:
            cite_status = "PASS" if citation_ok else "FAIL"
            print(f"    citation check: [{cite_status}] {citation_detail}")
        if result.get("citation_warnings"):
            print(f"    live citation warnings (verify_citations self-check):")
            for w in result["citation_warnings"]:
                print(f"      - {w}")

        results.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": answer,
            "passed": ok,
            "matched": hits,
            "citation_check_passed": citation_ok,
            "citation_check_detail": citation_detail,
            "citation_warnings": result.get("citation_warnings", []),
            "elapsed_seconds": round(elapsed, 1),
            "sources": [
                {"company": s["company"], "page": s["page"], "distance": round(s["distance"], 3)}
                for s in result["sources"]
            ],
        })

    total = len(EVAL_CASES)
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print(f"{'='*60}")

    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], {"passed": 0, "total": 0})
        by_category[r["category"]]["total"] += 1
        if r["passed"]:
            by_category[r["category"]]["passed"] += 1

    for cat, stats in by_category.items():
        print(f"  {cat}: {stats['passed']}/{stats['total']}")

    citation_checks = [r for r in results if r["citation_check_passed"] is not None]
    if citation_checks:
        citation_passed = sum(1 for r in citation_checks if r["citation_check_passed"])
        print(f"\n  Citation accuracy (single_fact only): {citation_passed}/{len(citation_checks)} "
              f"-- verifies the CITED source actually contains the figure, not just the answer text")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"passed": passed, "total": total, "by_category": by_category},
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    run_eval()