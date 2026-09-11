import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from extract import extract_pages
from chunk import chunk_page
from embed import embed_chunks, save_embeddings
import retriever
from generate import generate_answer

app = FastAPI(title="Study Buddy API")

# ---- Where each session's uploaded PDF + processed data lives ----
# Each session gets its own folder: sessions/<session_id>/upload.pdf,
# embeddings.npy, metadata.json. Keeping them in separate folders (instead of
# one shared embeddings.npy like before) is what makes sure session A's
# questions can never accidentally retrieve session B's document.
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

# ---- Serve the frontend ----
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


# ---- Request/response models ----
class AskRequest(BaseModel):
    session_id: str
    question: str


class Source(BaseModel):
    source: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    num_chunks: int


# ---- Health check ----
@app.get("/health")
def health():
    return {"status": "ok"}


# ---- Upload a PDF and process it into this session's own embeddings ----
@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # The browser sends the session_id it already has in localStorage. If
    # this is someone's very first upload ever, it won't have one yet - in
    # that case we mint a new one here and hand it back in the response.
    if not session_id:
        session_id = str(uuid.uuid4())

    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save the raw PDF to this session's folder before processing it - extract_pages()
    # needs a real file path to read from, not just the bytes in memory. We keep
    # the original filename (not e.g. "upload.pdf") so citations shown in the
    # chat UI say something meaningful like "biology_notes.pdf, page 3" instead
    # of a generic name. Path(...).name strips any directory components the
    # browser might send, so this can't be used to write outside session_dir.
    original_name = Path(file.filename).name
    pdf_path = session_dir / original_name
    with open(pdf_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    # Run the same extract -> chunk -> embed pipeline that run_pipeline.py and
    # embed.py use offline, just scoped to this one session's folder instead
    # of the fixed chunks.json/embeddings.npy files.
    try:
        pages = extract_pages(str(pdf_path))

        chunks = []
        for page in pages:
            chunks.extend(chunk_page(page))

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Couldn't find any text in that PDF (is it a scanned image?).",
            )

        # Give every chunk in this session a stable ID, same as run_pipeline.py does.
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = i

        embeddings = embed_chunks(chunks)
        save_embeddings(
            chunks,
            embeddings,
            session_dir / "embeddings.npy",
            session_dir / "metadata.json",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    # Drop any cached copy of this session's old data, so the next /ask call
    # reloads the fresh embeddings we just wrote instead of a stale version.
    retriever.forget_session(session_id)

    return UploadResponse(session_id=session_id, filename=file.filename, num_chunks=len(chunks))


# ---- Ask a question about THIS session's uploaded document ----
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    chunks = retriever.retrieve(question, request.session_id, top_k=3)

    if chunks is None:
        raise HTTPException(
            status_code=404,
            detail="No document found for this session. Please upload a PDF first.",
        )

    try:
        result = generate_answer(question, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    return AskResponse(answer=result["answer"], sources=result["sources"])
