"""
embed.py

Turns chunks.json (from run_pipeline.py) into vector embeddings.
Output:
  - embeddings.npy   : a NumPy array of shape (num_chunks, 384), one row per chunk
  - metadata.json    : the same chunk metadata, in the SAME ORDER as embeddings.npy,
                        so embeddings.npy[i] corresponds to metadata.json[i]

Why split into two files instead of one? .npy is a compact binary format built for
numeric arrays (fast to load), while .json is better for the human-readable metadata.
Keeping them as parallel arrays (matched by index) is a simple, common pattern for
small-to-medium projects before you'd reach for a real vector database.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_OUT = "embeddings.npy"
METADATA_OUT = "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(path):
    with open(path, "r") as f:
        return json.load(f)


def embed_chunks(chunks, model):
    # Pull out just the text to feed the model
    texts = [chunk["text"] for chunk in chunks]

    # model.encode() batches this efficiently rather than one-at-a-time.
    # show_progress_bar gives you a live progress indicator for larger note sets.
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def main():
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model '{MODEL_NAME}' (downloads once, then cached)...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding chunks...")
    embeddings = embed_chunks(chunks, model)

    print(f"Saving {embeddings.shape[0]} vectors of dimension {embeddings.shape[1]} to {EMBEDDINGS_OUT}")
    np.save(EMBEDDINGS_OUT, embeddings)

    with open(METADATA_OUT, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved matching metadata to {METADATA_OUT}")

    # --- Self-test: prove semantic search actually works on YOUR data ---
    print("\n--- Quick sanity check ---")
    test_query = input("Type a question about your notes to test retrieval (or press Enter to skip): ").strip()
    if test_query:
        query_vec = model.encode(test_query, convert_to_numpy=True)

        # Cosine similarity between the query and every chunk
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
        sims = (embeddings @ query_vec) / norms

        top_idx = np.argsort(sims)[::-1][:3]  # top 3 matches
        print(f"\nTop 3 chunks for: \"{test_query}\"\n")
        for rank, i in enumerate(top_idx, 1):
            c = chunks[i]
            print(f"{rank}. (score={sims[i]:.3f}) [{c.get('source')} p.{c.get('page')}]")
            print(f"   {c['text'][:150]}...\n")


if __name__ == "__main__":
    main()