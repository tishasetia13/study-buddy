"""
embed.py

Turns chunk dicts into vector embeddings.

This file is used two ways:
1. As a script (`python embed.py`) - the original offline workflow. Loads
   chunks.json (written by run_pipeline.py) and writes embeddings.npy +
   metadata.json, for manually pre-processing one fixed PDF.
2. As a module (`import embed`) - main.py's /upload endpoint calls
   embed_chunks() and save_embeddings() directly so each uploaded PDF can be
   embedded and saved into its own session folder, instead of always
   overwriting the one shared pair of files above.

Why .npy + .json as separate files? .npy is a compact binary format built for
numeric arrays (fast to load), while .json is better for human-readable
metadata. Keeping them as parallel arrays (matched by index) is a simple,
common pattern for small-to-medium projects before you'd reach for a real
vector database.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_OUT = "embeddings.npy"
METADATA_OUT = "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Loading the embedding model takes a few seconds, so we only want to do it
# once per process (not once per upload). Cache it at module level, same
# pattern retriever.py uses for the same reason.
_model = None


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_chunks(path):
    with open(path, "r") as f:
        return json.load(f)


def embed_chunks(chunks):
    """
    Turn a list of chunk dicts (each with a "text" field) into a NumPy array
    of embeddings, one row per chunk, in the same order as `chunks`.
    """
    model = _load_model()
    texts = [chunk["text"] for chunk in chunks]

    # model.encode() batches this efficiently rather than one-at-a-time.
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def save_embeddings(chunks, embeddings, embeddings_path, metadata_path):
    """
    Save embeddings + their matching chunk metadata to the given paths.
    Kept as two calls to the same file pair so a caller (like /upload) can
    point them at a per-session folder instead of the fixed files above.
    """
    np.save(embeddings_path, embeddings)
    with open(metadata_path, "w") as f:
        json.dump(chunks, f, indent=2)


def main():
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model '{MODEL_NAME}' (downloads once, then cached)...")
    embeddings = embed_chunks(chunks)

    print(f"Saving {embeddings.shape[0]} vectors of dimension {embeddings.shape[1]} to {EMBEDDINGS_OUT}")
    save_embeddings(chunks, embeddings, EMBEDDINGS_OUT, METADATA_OUT)
    print(f"Saved matching metadata to {METADATA_OUT}")


if __name__ == "__main__":
    main()
