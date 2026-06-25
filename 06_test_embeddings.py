"""
Step 0 (Phase 5): Confirm the local embedding model works.
Run this BEFORE building the full embedding pipeline.

This downloads the all-MiniLM-L6-v2 model the first time you run it
(~90MB, one-time download, cached locally afterward). No API key needed --
this runs entirely on your CPU.
"""

from sentence_transformers import SentenceTransformer

print("Loading embedding model (first run will download ~90MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.\n")

sentences = [
    "Net profit increased to $1,200 million this year.",
    "Revenue grew due to strong customer demand.",
    "The weather was sunny yesterday.",
]

embeddings = model.encode(sentences)

print(f"Embedded {len(sentences)} sentences.")
print(f"Each embedding has {embeddings.shape[1]} dimensions.")

# Sanity check: the two finance-related sentences should be more similar
# to each other than either is to the unrelated weather sentence.
from sentence_transformers.util import cos_sim

sim_finance_pair = cos_sim(embeddings[0], embeddings[1]).item()
sim_unrelated_pair = cos_sim(embeddings[0], embeddings[2]).item()

print(f"\nSimilarity between the two finance sentences: {sim_finance_pair:.3f}")
print(f"Similarity between a finance sentence and the weather sentence: {sim_unrelated_pair:.3f}")

if sim_finance_pair > sim_unrelated_pair:
    print("\nMakes sense: the finance sentences are more similar to each other.")
else:
    print("\nUnexpected: something may be wrong with the model.")
