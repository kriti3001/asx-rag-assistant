"""
Step 4 (Phase 5): Embed all chunks and store them in ChromaDB.

Loads every chunks/*_chunks.json file, embeds each chunk's text using the
local all-MiniLM-L6-v2 model, and stores the result in a persistent
ChromaDB collection on disk (chroma_db/). This is a one-time cost -- once
stored, ChromaDB persists across runs, so you only need to re-run this
if you change your chunking or add new companies.

Usage: python 07_embed_and_store.py
(processes every file in chunks/ automatically)
"""

import os
import glob
import json
import time

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "asx_annual_reports"
EMBEDDING_BATCH_SIZE = 64  # encode in batches for speed, rather than one at a time


def load_all_chunks(chunks_dir="chunks"):
    """Load every *_chunks.json file in the chunks directory."""
    all_chunks = []
    files = sorted(glob.glob(os.path.join(chunks_dir, "*_chunks.json")))
    if not files:
        raise FileNotFoundError(
            f"No chunk files found in '{chunks_dir}/'. Run 05_chunk_documents.py first."
        )
    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            company_chunks = json.load(f)
        print(f"  Loaded {len(company_chunks)} chunks from {os.path.basename(filepath)}")
        all_chunks.extend(company_chunks)
    return all_chunks


def flatten_metadata(metadata):
    """
    ChromaDB metadata values must be str/int/float/bool (no nested dicts).
    Our chunk metadata is already flat, but we ensure all values are of
    an allowed type here defensively, since a stray None or list would
    cause ChromaDB to reject the whole batch.
    """
    clean = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean


def main():
    print("Loading chunks from disk...")
    chunks = load_all_chunks()
    print(f"\nTotal chunks to embed: {len(chunks)}")

    print("\nLoading embedding model (cached after first run)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nInitializing ChromaDB (persistent, stored in chroma_db/)...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # If a collection from a previous run exists, drop it so we start fresh
    # rather than silently appending duplicate entries on re-runs.
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Existing collection '{COLLECTION_NAME}' found -- deleting before rebuilding.")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    print(f"\nEmbedding and storing {len(chunks)} chunks in batches of {EMBEDDING_BATCH_SIZE}...")
    t0 = time.time()

    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        # Embed using embedding_text (company name prepended -- see
        # 05_chunk_documents.py for why), but store the original clean
        # "text" as the document, so retrieval results and anything shown
        # to the user/LLM later stay exactly as extracted, with no
        # "COMPANY: " prefix artifact. Falls back to "text" for any older
        # chunk files that pre-date the embedding_text field.
        display_texts = [c["text"] for c in batch]
        embed_texts = [c.get("embedding_text", c["text"]) or c["text"] for c in batch]
        # Defensive: a blank embedding_text would make encode() produce a
        # meaningless all-near-zero vector rather than error outright, so
        # we guard against it explicitly rather than relying on silence.
        for idx, t in enumerate(embed_texts):
            if not t.strip():
                embed_texts[idx] = display_texts[idx] or "(empty chunk)"
        ids = [c["chunk_id"] for c in batch]
        metadatas = [flatten_metadata(c["metadata"]) for c in batch]

        try:
            embeddings = model.encode(embed_texts, show_progress_bar=False).tolist()
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=display_texts,
                metadatas=metadatas,
            )
        except Exception as e:
            done_so_far = i
            raise RuntimeError(
                f"Embedding failed on batch starting at chunk {i} "
                f"(chunk_ids {ids[0]}..{ids[-1]}). {done_so_far} chunks were "
                f"successfully embedded and stored before this failure -- "
                f"they remain in the ChromaDB collection. Original error: {e}"
            )

        done = min(i + EMBEDDING_BATCH_SIZE, len(chunks))
        elapsed = time.time() - t0
        print(f"  {done}/{len(chunks)} chunks embedded ({elapsed:.0f}s elapsed)")

    total_time = time.time() - t0
    print(f"\nDone. Embedded and stored {len(chunks)} chunks in {total_time:.0f} seconds.")
    print(f"ChromaDB collection '{COLLECTION_NAME}' saved to '{CHROMA_DB_PATH}/'.")
    print(f"Collection now contains {collection.count()} items.")


if __name__ == "__main__":
    main()