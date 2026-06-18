"""
PDF ingestion and text chunking pipeline.

Reads PDFs from the books/ directory, extracts text page-by-page,
splits into overlapping chunks, generates embeddings, and stores
everything in Qdrant (local embedded vector database).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader

from app.config import settings
from app.embeddings import encode_chunks
from app.vectordb import clear_book, clear_collection, insert_chunks

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────────


def ingest_all_books() -> dict:
    """Run the full ingestion pipeline for every PDF in the books/ directory.

    Returns a result dict suitable for IngestResponse.
    """
    start = time.perf_counter()
    books_dir = settings.books_dir

    if not books_dir.exists():
        raise FileNotFoundError(f"Books directory not found: {books_dir}")

    pdf_paths = sorted(books_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {books_dir}")

    logger.info("Starting ingestion of %d PDF(s) …", len(pdf_paths))

    # ── Wipe the entire collection for a clean rebuild ──────────────────
    # This removes any stale entries (e.g. old book1/book2 test data)
    # so that only currently-present PDFs exist in the vector store.
    logger.info("Clearing existing collection for a fresh rebuild …")
    clear_collection()
    logger.info("Books processed: %d", len(pdf_paths))

    total_chunks = 0
    errors: List[str] = []

    for pdf_path in pdf_paths:
        try:
            chunks_count = _ingest_one_book(pdf_path)
            total_chunks += chunks_count
            logger.info(
                "✓ '%s' — %d chunks stored.",
                pdf_path.name,
                chunks_count,
            )
        except Exception as exc:
            msg = f"✗ '{pdf_path.name}': {exc}"
            logger.error(msg)
            errors.append(msg)

    elapsed = time.perf_counter() - start
    status = "success" if not errors else ("partial" if total_chunks > 0 else "error")

    logger.info(
        "Ingestion complete: %d chunks across %d books in %.1fs.",
        total_chunks,
        len(pdf_paths) - len([e for e in errors if "No PDF" not in e]),
        elapsed,
    )

    return {
        "status": status,
        "books_processed": len(pdf_paths) - len(errors),
        "total_chunks": total_chunks,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 2),
    }


# ── Internal Helpers ─────────────────────────────────────────────────────


def _ingest_one_book(pdf_path: Path) -> int:
    """Ingest a single PDF: extract → chunk → embed → store.

    Returns the number of chunks stored.
    """
    book_name = pdf_path.stem  # filename without .pdf extension

    # 1. Extract text from every page
    pages: List[Tuple[int, str]] = _extract_pages(pdf_path)
    if not pages:
        logger.warning("No extractable text in '%s'.", pdf_path.name)
        return 0

    # 2. Split into overlapping chunks
    chunks: List[Tuple[str, int, str, str]] = _chunk_pages(book_name, pages)

    # 3. Generate embeddings in batch
    chunk_texts = [c[2] for c in chunks]
    embeddings = encode_chunks(chunk_texts)

    # 4. Clear old data for this book (idempotent re-ingestion)
    clear_book(book_name)

    # 5. Bulk insert
    rows: List[Tuple[str, int, str, str, List[float]]] = [
        (chunks[i][0], chunks[i][1], chunks[i][2], chunks[i][3], embeddings[i])
        for i in range(len(chunks))
    ]
    insert_chunks(rows)

    return len(rows)


def _extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract text from every page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of (page_number, page_text).  Pages with no extractable
        text are skipped.
    """
    pages: List[Tuple[int, str]] = []
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Cannot read PDF: {exc}") from exc

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            # Try to extract the actual printed book page number from the text.
            # Falls back to PDF page index if not found.
            book_page_num = _extract_book_page_number(text, page_idx)
            # Normalise whitespace: collapse newlines and multiple spaces.
            clean = _normalise_text(text)
            if clean:
                pages.append((book_page_num, clean))

    logger.debug("Extracted %d non-empty pages from '%s'.", len(pages), pdf_path.name)
    return pages


def _chunk_pages(
    book_name: str,
    pages: List[Tuple[int, str]],
) -> List[Tuple[str, int, str, str]]:
    """Split pages into overlapping word-count-based chunks.

    Chunking strategy:
      - Split each page's text into words.
      - Slide a window of `chunk_size_words` words with a step
        of `chunk_size_words - chunk_overlap_words`.
      - Each chunk carries the book_name and page_number of its
        starting page.

    Args:
        book_name: Name of the source book.
        pages: List of (page_number, page_text).

    Returns:
        List of (book_name, page_number, chunk_text, chapter) tuples.
    """
    chunk_size = settings.chunk_size_words
    overlap = settings.chunk_overlap_words
    step = max(chunk_size - overlap, 1)

    chunks: List[Tuple[str, int, str, str]] = []

    for page_num, page_text in pages:
        words = page_text.split()
        if not words:
            continue

        # If the page is shorter than one chunk, emit as a single chunk.
        if len(words) <= chunk_size:
            chunk_text = " ".join(words)
            chapter = _detect_chapter(chunk_text)
            chunks.append((book_name, page_num, chunk_text, chapter))
            continue

        # Slide the window
        start = 0
        while start < len(words):
            window = words[start : start + chunk_size]
            chunk_text = " ".join(window)
            chapter = _detect_chapter(chunk_text)
            chunks.append((book_name, page_num, chunk_text, chapter))
            start += step

    logger.debug(
        "Book '%s': %d chunks from %d pages (size=%d, overlap=%d).",
        book_name,
        len(chunks),
        len(pages),
        chunk_size,
        overlap,
    )
    return chunks


def _detect_chapter(text: str) -> str:
    """Try to detect a chapter name or number from chunk text.

    Looks for patterns like:
      - 'Chapter 1', 'Chapter - 1', 'CHAPTER ONE'
      - 'Section 2', 'Part 3'
      - Lines that look like headings (short, all-caps or title-case)

    Returns an empty string if nothing is found.
    """
    # Pattern: Chapter X or Chapter - X or Chapter: X
    match = re.search(
        r'(?i)\b(chapter\s*[-:—]?\s*\d+[^\n\.]{0,60})',
        text
    )
    if match:
        return match.group(1).strip()[:80]

    # Pattern: Section X or Part X
    match = re.search(
        r'(?i)\b((?:section|part)\s*[-:—]?\s*\d+[^\n\.]{0,60})',
        text
    )
    if match:
        return match.group(1).strip()[:80]

    return ""


def _extract_book_page_number(text: str, fallback: int) -> int:
    """Extract the actual printed page number from PDF page text.

    Many spiritual books print the page number at the very start or end
    of the page text. This function looks for a standalone number at the
    beginning or end of the raw text before normalisation.

    Examples it handles:
      - "69 Pleasure Chapter - 6 When we were kids..."  → 69
      - "— 3 — 7 Q: Is suffering..."                    → 3
      - "59 What Is Happiness?..."                       → 59
      - "5 For 25 years..."                              → 5

    Args:
        text: Raw extracted page text (before normalisation).
        fallback: PDF page index to use if no number is found.

    Returns:
        The detected book page number, or the fallback PDF page index.
    """
    stripped = text.strip()

    # Pattern 1: Number at very start of text (e.g. "69 Pleasure Chapter")
    match = re.match(r'^(\d{1,4})\s+\D', stripped)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 9999:
            return num

    # Pattern 2: Number wrapped in dashes at start (e.g. "— 3 —" or "- 3 -")
    match = re.match(r'^[\—\-–]\s*(\d{1,4})\s*[\—\-–]', stripped)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 9999:
            return num

    # Pattern 3: Number at very end of text (footer page numbers)
    match = re.search(r'\s(\d{1,4})\s*$', stripped)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 9999:
            return num

    return fallback


def _normalise_text(text: str) -> str:
    """Clean extracted PDF text.

    - Collapse multiple whitespace characters into a single space.
    - Remove common PDF extraction artefacts (hyphenated line-breaks).
    - Strip leading/trailing whitespace.
    """
    # Rejoin hyphenated words split across lines: "teach-\nings" → "teachings"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse all whitespace (newlines, tabs, multiple spaces) to a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Standalone Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    print("Starting ingestion pipeline …")
    try:
        result = ingest_all_books()
        print("\n" + "-" * 40)
        print(f"Status          : {result['status']}")
        print(f"Books processed : {result['books_processed']}")
        print(f"Total chunks    : {result['total_chunks']}")
        print(f"Time taken      : {result['elapsed_seconds']}s")
        if result["errors"]:
            print("Errors:")
            for e in result["errors"]:
                print(f"  {e}")
        print("-" * 40)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
