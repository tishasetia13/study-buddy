"""
STEP 1 (full): Point this at a folder of PDFs -> get back a JSON file of
clean, chunked, metadata-tagged text, ready for the next step (embeddings).

Usage:
    python run_pipeline.py

Change PDF_FOLDER below to wherever your PDFs live.
"""

import json
from pathlib import Path

from extract import extract_pages
from chunk import chunk_page

PDF_FOLDER = "."                       # folder containing your PDFs
OUTPUT_FILE = "chunks.json"            # where the chunked output goes


def process_pdf(pdf_path: str) -> list[dict]:
    pages = extract_pages(pdf_path)
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_page(page))
    return all_chunks


def main():
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {PDF_FOLDER}/ — put some PDFs there and re-run.")
        return

    all_chunks = []
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        chunks = process_pdf(str(pdf_file))
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    # Give every chunk a global, stable ID. We'll need this in the next
    # step to map embeddings back to their source text.
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nDone. {len(all_chunks)} total chunks written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()