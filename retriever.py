"""
retriever.py

Given a question, find the most relevant chunks of text to feed the LLM.

Before multi-user support, this loaded ONE global embeddings.npy/metadata.json
pair at startup and every question searched that same fixed index. Now each
browser session has its own uploaded PDF, so we load a *different*
embeddings.npy/metadata.json pair per session_id, from that session's own
folder under sessions/. This is what keeps one person's document from ever
showing up in another person's answers.
"""

import json
import numpy as np
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
SESSIONS_DIR = Path("sessions")

# Loading the embedding model takes a few seconds, so we cache it once per
# process rather than reloading it on every question.
_model = None

# Cache of already-loaded sessions, keyed by session_id, so we don't hit disk
# on every single question: {session_id: {"embeddings": ..., "metadata": ...}}
_session_cache = {}


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _session_paths(session_id):
    session_dir = SESSIONS_DIR / session_id
    return session_dir / "embeddings.npy", session_dir / "metadata.json"


def _load_session(session_id):
    """Read one session's embeddings+metadata off disk and cache them in memory."""
    embeddings_path, metadata_path = _session_paths(session_id)

    if not embeddings_path.exists() or not metadata_path.exists():
        return None

    embeddings = np.load(embeddings_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    session_data = {"embeddings": embeddings, "metadata": metadata}
    _session_cache[session_id] = session_data
    return session_data


def forget_session(session_id):
    """
    Drop a session's cached embeddings from memory.

    Call this right after a fresh upload finishes writing new files to disk,
    so the NEXT question for that session reloads the new data instead of
    reusing a stale copy left over from an earlier upload.
    """
    _session_cache.pop(session_id, None)


def _cosine_similarity(query_vec, all_vecs):
    query_norm = query_vec / np.linalg.norm(query_vec)
    all_norm = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return all_norm @ query_norm


def retrieve(query, session_id, top_k=3):
    """
    Given a question and a session_id, return the top_k most relevant chunks
    from THAT session's document only.

    Returns a list of dicts, best match first:
    [{"text": ..., "source": ..., "page": ..., "score": ...}, ...]

    Returns None if this session has no processed document yet (e.g. nobody
    has uploaded a PDF for it, or the server restarted and lost it).
    """
    session_data = _session_cache.get(session_id)
    if session_data is None:
        session_data = _load_session(session_id)
    if session_data is None:
        return None

    model = _load_model()
    query_vec = model.encode(query)
    scores = _cosine_similarity(query_vec, session_data["embeddings"])
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk_meta = session_data["metadata"][idx]
        results.append({
            "text": chunk_meta["text"],
            "source": chunk_meta["source"],
            "page": chunk_meta["page"],
            "score": float(scores[idx]),
        })
    return results
