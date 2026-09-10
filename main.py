from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from retriever import retrieve
from generate import generate_answer

app = FastAPI(title="Study Buddy API")


# ---- Request model: the order form ----
class AskRequest(BaseModel):
    question: str


# ---- Response model: the receipt ----
class Source(BaseModel):
    source: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


# ---- Health check: "is the restaurant even open?" ----
@app.get("/health")
def health():
    return {"status": "ok"}


# ---- The real endpoint ----
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    chunks = retrieve(question, top_k=3)

    try:
        result = generate_answer(question, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    return AskResponse(answer=result["answer"], sources=result["sources"])