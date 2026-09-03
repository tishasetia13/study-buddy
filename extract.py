"""
STEP 1a: Extract text from PDFs, page by page.

Why page-by-page instead of one big string?
Because later, when the chatbot answers a question, we want to say
"this came from page 12 of biology_notes.pdf" - not just "somewhere
in your PDF". Keeping page numbers attached to text is what makes
citations possible.

NOTE: this simple version reads the full page width at once, so it
works well for single-column PDFs (like your class notes) but will
garble two-column academic papers (columns get read as if they were
one line). Fine for now - we'll only need the column-aware version
if/when you feed it a two-column paper.
"""

import pdfplumber
from pathlib import Path


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Read a PDF and return a list of dicts, one per page:
    [{"source": "notes.pdf", "page": 1, "text": "..."}, ...]
    """
    pdf_path = Path(pdf_path)
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text()  # returns None if page has no extractable text (e.g. a scanned image)

            if raw_text is None:
                # This happens with scanned/image-only PDFs. Flagging it now
                # rather than silently skipping - you'd need OCR (later problem) for these.
                print(f"  Warning: no extractable text on page {i} of {pdf_path.name} (scanned image?)")
                continue

            pages.append({
                "source": pdf_path.name,
                "page": i,
                "text": raw_text
            })

    return pages


if __name__ == "__main__":
    # Quick manual test
    pages = extract_pages("sample_notes.pdf")
    print(f"\nExtracted {len(pages)} pages from sample_notes.pdf\n")
    for p in pages:
        print(f"--- Page {p['page']} ---")
        print(p["text"][:150], "...\n")
 
