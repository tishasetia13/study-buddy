import json
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "embeddings.npy"
METADATA_PATH = "metadata.json"

_model = None
_embeddings = None
_metadata = None


def _load_resources():
    global _model, _embeddings, _metadata

    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)

    if _embeddings is None:
        _embeddings = np.load(EMBEDDINGS_PATH)

    if _metadata is None:
        with open(METADATA_PATH, "r") as f:
            _metadata = json.load(f)


def _cosine_similarity(query_vec, all_vecs):
    query_norm = query_vec / np.linalg.norm(query_vec)
    all_norm = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return all_norm @ query_norm


def retrieve(query, top_k=3):
    """
    Given a question, return the top_k most relevant chunks.
    Returns a list of dicts, best match first:
    [{"text": ..., "source": ..., "page": ..., "score": ...}, ...]
    """
    _load_resources()
    query_vec = _model.encode(query)
    scores = _cosine_similarity(query_vec, _embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk_meta = _metadata[idx]
        results.append({
            "text": chunk_meta["text"],
            "source": chunk_meta["source"],
            "page": chunk_meta["page"],
            "score": float(scores[idx]),
        })
    return results


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or 'quit'): ")
        if query.lower() == "quit":
            break
        results = retrieve(query)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. [{r['source']} p.{r['page']}] score={r['score']:.3f}")
            print(r["text"][:200])