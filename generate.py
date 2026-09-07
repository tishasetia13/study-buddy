"""
Step 4: LLM generation.

Takes chunks from retriever.py, builds a grounded prompt (answer ONLY from the
provided excerpts), calls Gemini, and returns an answer with citations mapped
back to real source/page numbers from our own metadata.
"""
import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

MODEL_NAME = "gemini-flash-latest"

SYSTEM_INSTRUCTION = """You are a study assistant. You answer questions using ONLY the excerpts \
provided below — never your own outside knowledge, even if you are confident about the answer.

Rules:
1. Every claim you make must be supported by one of the excerpts.
2. After each claim, tag it with the excerpt(s) it came from, like this: [Excerpt 1]
3. If the excerpts do not contain enough information to answer the question, say exactly: \
"I don't have enough information in your notes to answer that." Do not guess or fill gaps \
with outside knowledge.
4. Be concise. Do not repeat the excerpts back verbatim — synthesize them in your own words."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Turn retriever output into a single prompt string with labeled excerpts.

    chunks: list of {"text", "source", "page", "score"} from retriever.retrieve()
    """
    excerpt_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        block = f"[Excerpt {i} — {chunk['source']}, page {chunk['page']}]\n{chunk['text']}"
        excerpt_blocks.append(block)

    excerpts_text = "\n\n".join(excerpt_blocks)

    prompt = f"""{SYSTEM_INSTRUCTION}

Excerpts:

{excerpts_text}

Question: {query}

Answer:"""
    return prompt


def parse_citations(answer_text: str, chunks: list[dict]) -> list[dict]:
    """
    Find every [Excerpt N] tag the model used and map it back to the real
    source/page from our own metadata (never trust the model's own memory
    of page numbers — only ours).
    """
    used_indices = sorted(set(int(n) for n in re.findall(r"\[Excerpt (\d+)\]", answer_text)))
    sources = []
    for idx in used_indices:
        if 1 <= idx <= len(chunks):
            chunk = chunks[idx - 1]
            sources.append({"source": chunk["source"], "page": chunk["page"]})
    return sources


@retry(
    retry=retry_if_exception_type(ServerError),  # only retry on server-side errors (5xx), not our own bugs
    stop=stop_after_attempt(4),                   # try up to 4 times total
    wait=wait_exponential(multiplier=1, min=2, max=20),  # wait 2s, 4s, 8s... between tries
    reraise=True,                                  # if all 4 attempts fail, raise the real error (don't hide it)
)
def _call_gemini(client, prompt: str):
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"temperature": 0.2},
    )


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Full Step 4 pipeline: build prompt -> call Gemini -> parse citations.
    Returns {"answer": str, "sources": [{"source", "page"}, ...]}
    """
    if not chunks:
        return {
            "answer": "I don't have enough information in your notes to answer that.",
            "sources": [],
        }

    prompt = build_prompt(query, chunks)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = _call_gemini(client, prompt)

    answer_text = response.text
    sources = parse_citations(answer_text, chunks)

    return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    # Quick manual test against the real API using retriever.py output.
    from retriever import retrieve

    query = "What is the boiling point of mercury in Fahrenheit?"
    chunks = retrieve(query, top_k=3)
    result = generate_answer(query, chunks)

    print("ANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - {s['source']}, page {s['page']}")