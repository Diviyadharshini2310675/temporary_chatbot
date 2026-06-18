"""
DOCX Q&A Import Script.

Reads Question & Answer pairs from a DOCX file and imports them
into the PostgreSQL qa_dataset table.

Usage:
    python import_qa_docx.py --file path/to/qa_file.docx
    python import_qa_docx.py --file path/to/qa_file.docx --source-book "Book Name"

Requirements:
    pip install python-docx psycopg2-binary python-dotenv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from docx import Document

# ── Load .env ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("import_qa_docx")


# ── Database connection ───────────────────────────────────────────────────

def get_db_conn():
    """Create and return a raw psycopg2 connection."""
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        dbname=os.getenv("PG_DBNAME", "spiritual_chatbot"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
    )


# ── DOCX Parsing ─────────────────────────────────────────────────────────

def parse_qa_pairs(docx_path: Path) -> list[dict]:
    """Extract Q&A pairs from a DOCX file.

    Supports two formats:
      Format A — Labelled paragraphs:
        Q: What is the soul?
        A: The soul is eternal.

      Format B — Alternating paragraphs (odd = question, even = answer)
        What is the soul?
        The soul is eternal.

    Returns:
        List of dicts with keys: question, answer.
    """
    doc = Document(str(docx_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    qa_pairs = []

    # Try Format A first — look for Q:/A: labels
    q_pattern = re.compile(r"^[Qq]\s*[:\.\)]\s*(.+)", re.DOTALL)
    a_pattern = re.compile(r"^[Aa]\s*[:\.\)]\s*(.+)", re.DOTALL)

    i = 0
    found_labeled = False
    while i < len(paragraphs):
        q_match = q_pattern.match(paragraphs[i])
        if q_match and i + 1 < len(paragraphs):
            a_match = a_pattern.match(paragraphs[i + 1])
            if a_match:
                qa_pairs.append({
                    "question": q_match.group(1).strip(),
                    "answer": a_match.group(1).strip(),
                })
                found_labeled = True
                i += 2
                continue
        i += 1

    # If labeled format found something, return it
    if found_labeled and qa_pairs:
        logger.info("Parsed %d Q&A pairs using labeled format (Q:/A:).", len(qa_pairs))
        return qa_pairs

    # Fallback: alternating paragraphs
    qa_pairs = []
    for i in range(0, len(paragraphs) - 1, 2):
        qa_pairs.append({
            "question": paragraphs[i],
            "answer": paragraphs[i + 1],
        })
    logger.info("Parsed %d Q&A pairs using alternating paragraph format.", len(qa_pairs))
    return qa_pairs


# ── Import Logic ──────────────────────────────────────────────────────────

def import_to_db(
    qa_pairs: list[dict],
    source_book: str = "",
    source_page: int = 0,
) -> dict:
    """Insert Q&A pairs into qa_dataset using UPSERT with MD5 hash deduplication.

    Uses MD5 hash of normalized_question for the unique index to avoid
    PostgreSQL btree index size limits on long text values.

    Args:
        qa_pairs: List of {question, answer} dicts.
        source_book: Optional book name to tag each row.
        source_page: Optional page number to tag each row.

    Returns:
        Dict with total_found, inserted, skipped.
    """
    inserted = 0
    skipped = 0
    total = len(qa_pairs)

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for pair in qa_pairs:
                question = pair["question"].strip()
                answer = pair["answer"].strip()

                if not question or not answer:
                    skipped += 1
                    continue

                normalized = question.lower()
                question_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()

                cur.execute(
                    """
                    INSERT INTO qa_dataset
                        (question, normalized_question, answer, source_book, source_page, question_hash)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (question_hash) DO NOTHING
                    """,
                    (question, normalized, answer, source_book, source_page, question_hash),
                )

                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1

        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Database insert failed: %s", exc)
        raise
    finally:
        conn.close()

    return {"total_found": total, "inserted": inserted, "skipped": skipped}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Import Q&A pairs from DOCX into PostgreSQL.")
    parser.add_argument("--file", required=True, help="Path to the DOCX file.")
    parser.add_argument("--source-book", default="", help="Book name to tag rows with.")
    parser.add_argument("--source-page", type=int, default=0, help="Page number to tag rows with.")
    args = parser.parse_args()

    docx_path = Path(args.file)
    if not docx_path.exists():
        logger.error("File not found: %s", docx_path)
        sys.exit(1)

    logger.info("Parsing DOCX: %s", docx_path)
    qa_pairs = parse_qa_pairs(docx_path)

    if not qa_pairs:
        logger.warning("No Q&A pairs found in the file.")
        sys.exit(0)

    logger.info("Importing %d pairs into PostgreSQL …", len(qa_pairs))
    stats = import_to_db(qa_pairs, source_book=args.source_book, source_page=args.source_page)

    print("\n" + "-" * 35)
    print(f"Total Q&A Found:      {stats['total_found']}")
    print(f"Inserted:             {stats['inserted']}")
    print(f"Duplicates Skipped:   {stats['skipped']}")
    print("Import Completed Successfully")
    print("-" * 35 + "\n")


if __name__ == "__main__":
    main()
