"""
MySQL Q&A service.

Provides:
  - init_cache_table()             → create chatbot_cache table if not exists
  - get_cached_response(question)  → CACHE HIT: return stored answer + sources
  - save_cached_response(...)      → CACHE SAVE: store answer + sources
  - find_qa_answer(question)       → look up qa_dataset for pre-stored answers
  - save_user_question(question)   → store question in user_questions table
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from app.db import get_conn

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────


def _normalize_question(question: str) -> str:
    """Normalize a question for consistent hashing.

    Steps:
      1. Lowercase
      2. Strip leading/trailing whitespace
      3. Collapse multiple spaces into one
      4. Remove spaces before punctuation (? ! . , : ;)

    Examples:
      "Why humans suffer?"    → "why humans suffer?"
      "why humans suffer ?"   → "why humans suffer?"
      " WHY HUMANS SUFFER ? " → "why humans suffer?"
    """
    import re
    text = question.lower().strip()
    text = re.sub(r'\s+', ' ', text)              # collapse multiple spaces
    text = re.sub(r'\s+([?!.,;:])', r'\1', text)  # remove space before punctuation
    return text


def _hash_question(question: str) -> str:
    """Return MD5 hash of normalized question."""
    return hashlib.md5(_normalize_question(question).encode("utf-8")).hexdigest()


# ── Chatbot Cache ─────────────────────────────────────────────────────────


def init_cache_table() -> None:
    """Create chatbot_cache table in MySQL if it does not exist.

    Called once on application startup.
    """
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chatbot_cache (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    question_hash VARCHAR(32)  NOT NULL,
                    question      TEXT         NOT NULL,
                    answer        TEXT         NOT NULL,
                    sources       JSON         NOT NULL,
                    usage_count   INT          DEFAULT 1,
                    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_question_hash (question_hash)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            cur.close()
        logger.info("chatbot_cache table ready.")
    except Exception as exc:
        logger.error("Failed to create chatbot_cache table: %s", exc)


def get_cached_response(question: str) -> Optional[dict]:
    """Check chatbot_cache for a previously answered question.

    On cache hit:
      - Returns dict with 'answer' and 'sources'.
      - Increments usage_count.
      - Logs CACHE HIT.

    On cache miss:
      - Returns None.
      - Logs CACHE MISS.

    Args:
        question: Raw user question string.
    """
    if not question or not question.strip():
        return None

    q_hash = _hash_question(question)

    try:
        with get_conn() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT answer, sources FROM chatbot_cache "
                "WHERE question_hash = %s LIMIT 1",
                (q_hash,),
            )
            row = cur.fetchone()
            cur.close()

            if row:
                # Increment usage_count
                cur2 = conn.cursor()
                cur2.execute(
                    "UPDATE chatbot_cache SET usage_count = usage_count + 1 "
                    "WHERE question_hash = %s",
                    (q_hash,),
                )
                cur2.close()
                logger.info("CACHE HIT for question: '%s'", question[:80])
                # MySQL JSON column returns a string — parse it
                sources = row["sources"]
                if isinstance(sources, str):
                    sources = json.loads(sources)
                return {"answer": row["answer"], "sources": sources}
    except Exception as exc:
        logger.error("get_cached_response failed: %s", exc)

    logger.info("CACHE MISS for question: '%s'", question[:80])
    return None


def save_cached_response(
    question: str,
    answer: str,
    sources: list,
) -> None:
    """Save question + answer + sources to chatbot_cache.

    Uses INSERT IGNORE to prevent duplicates (same as ON CONFLICT DO NOTHING).

    Args:
        question: Raw user question.
        answer: Generated answer text.
        sources: List of source dicts (JSON-serializable).
    """
    if not question or not answer:
        return

    q_hash = _hash_question(question)

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT IGNORE INTO chatbot_cache
                    (question_hash, question, answer, sources)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    q_hash,
                    question.strip(),
                    answer,
                    json.dumps(sources),
                ),
            )
            cur.close()
        logger.info("CACHE SAVED for question: '%s'", question[:80])
    except Exception as exc:
        logger.error("save_cached_response failed: %s", exc)


# ── Q&A Dataset ───────────────────────────────────────────────────────────


def find_qa_answer(question: str) -> Optional[str]:
    """Search qa_dataset for a pre-stored answer (exact hash match).

    Args:
        question: Raw user question string.

    Returns:
        Stored answer string if found, else None.
    """
    if not question or not question.strip():
        return None

    q_hash = _hash_question(question)

    try:
        with get_conn() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT answer FROM qa_dataset "
                "WHERE question_hash = %s LIMIT 1",
                (q_hash,),
            )
            row = cur.fetchone()
            cur.close()
            if row:
                logger.info("QA dataset hit for question: '%s'", question[:80])
                return row["answer"]
    except Exception as exc:
        logger.error("find_qa_answer failed: %s", exc)

    return None


def save_user_question(question: str) -> None:
    """Save user question to user_questions table (no duplicates).

    Uses INSERT IGNORE to skip duplicates.

    Args:
        question: Raw user question string.
    """
    if not question or not question.strip():
        return

    normalized = question.strip().lower()

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT IGNORE INTO user_questions (question, normalized_question)
                VALUES (%s, %s)
                """,
                (question.strip(), normalized),
            )
            cur.close()
        logger.debug("Saved user question: '%s'", question[:80])
    except Exception as exc:
        logger.error("save_user_question failed: %s", exc)
