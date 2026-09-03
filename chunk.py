"""
STEP 1b: Split extracted page text into overlapping, semantically-coherent chunks.

Strategy: recursive splitting.
Try splitting on the "biggest" structural boundary first (paragraphs).
If a piece is still too big, fall back to a smaller boundary (sentences),
then to words if desperate. This avoids severing sentences mid-thought,
which fixed-size "cut every 500 characters" chunking does not avoid.

We also add OVERLAP between consecutive chunks so context isn't lost
right at a chunk boundary.
"""

import re

# These are tunable. Rough rule of thumb for a study assistant:
# - Big enough to hold a full idea/paragraph (better retrieval quality)
# - Small enough to not waste LLM context on irrelevant text
CHUNK_SIZE = 800       # target size in characters
CHUNK_OVERLAP = 150    # characters repeated between consecutive chunks


def split_into_paragraphs(text: str) -> list[str]:
    # Paragraphs are usually separated by blank lines. Fall back to
    # single newlines if the PDF extraction didn't preserve blank lines.
    paras = re.split(r"\n\s*\n", text)
    if len(paras) == 1:
        paras = text.split("\n")
    return [p.strip() for p in paras if p.strip()]


def split_into_sentences(text: str) -> list[str]:
    # Simple sentence splitter on '.', '!', '?' followed by a space + capital letter.
    # Not perfect (breaks on abbreviations like "Dr." sometimes) but good enough
    # for a first working version - you can swap in a proper NLP sentence
    # tokenizer (e.g. nltk) later if this bites you.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def recursive_split(text: str, max_size: int) -> list[str]:
    """Split text into pieces no larger than max_size, preferring paragraph
    then sentence then word boundaries."""
    if len(text) <= max_size:
        return [text]

    # Try paragraph-level split first
    pieces = split_into_paragraphs(text)
    if len(pieces) == 1:
        # No paragraph breaks found - drop to sentence level
        pieces = split_into_sentences(text)
    if len(pieces) == 1:
        # Still one piece (e.g. a wall of text with no punctuation) - split on words
        words = text.split()
        pieces = [" ".join(words[i:i + 50]) for i in range(0, len(words), 50)]

    # Recursively split any piece that's still too big
    result = []
    for piece in pieces:
        if len(piece) > max_size:
            result.extend(recursive_split(piece, max_size))
        else:
            result.append(piece)
    return result


def merge_with_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack small pieces (sentences/paragraphs) together up to
    chunk_size, then start the next chunk by repeating the tail of the
    previous one (the overlap)."""
    chunks = []
    current = ""

    for piece in pieces:
        if len(current) + len(piece) + 1 <= chunk_size:
            current = (current + " " + piece).strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, seeded with overlap from the end of the last chunk
            tail = current[-overlap:] if current else ""
            current = (tail + " " + piece).strip()

    if current:
        chunks.append(current)

    return chunks


def chunk_page(page: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Take one page dict {source, page, text} and return a list of chunk dicts."""
    pieces = recursive_split(page["text"], chunk_size)
    merged = merge_with_overlap(pieces, chunk_size, overlap)

    chunks = []
    for i, chunk_text in enumerate(merged):
        chunks.append({
            "source": page["source"],
            "page": page["page"],
            "chunk_index_on_page": i,
            "text": chunk_text,
            "char_count": len(chunk_text),
        })
    return chunks


if __name__ == "__main__":
    from extract import extract_pages

    pages = extract_pages("sample_notes.pdf")
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_page(page))

    print(f"Produced {len(all_chunks)} chunks from {len(pages)} pages\n")
    for c in all_chunks:
        print(f"[{c['source']} p.{c['page']} chunk {c['chunk_index_on_page']}] ({c['char_count']} chars)")
        print(c["text"][:120], "...\n")