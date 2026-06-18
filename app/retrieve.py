"""
Semantic retrieval module.

Takes a user question, generates its embedding, and fetches the
most relevant document chunks from Qdrant via cosine similarity.
"""

from __future__ import annotations

import logging
from typing import List

from app.config import settings
from app.embeddings import encode_query
from app.models import RetrievalResult
from app.vectordb import similarity_search

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    question: str,
    top_k: int | None = None,
) -> List[RetrievalResult]:
    """Retrieve the most relevant chunks for a user question.

    Pipeline:
      1. Encode the question into an embedding vector.
      2. Query Qdrant for the top_k * 3 most similar chunks (wider net).
      3. Deduplicate by chunk_text — keep only the highest-scoring copy.
      4. Return the top_k unique results sorted by descending similarity.

    Args:
        question: The user's raw question string.
        top_k: Number of chunks to return (default from settings).

    Returns:
        List of RetrievalResult, sorted by descending similarity, deduplicated.

    Raises:
        ValueError: If the question is empty.
        RuntimeError: If the database query fails.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    if top_k is None:
        top_k = settings.top_k_results

    logger.info("Retrieving top-%d chunks for question: '%s'", top_k, question[:100])

    # 1. Embed the question
    try:
        query_vec = encode_query(question.strip())
    except Exception as exc:
        logger.error("Failed to encode question: %s", exc)
        raise RuntimeError(f"Embedding generation failed: {exc}") from exc

    # 2. Vector similarity search — fetch more candidates to allow deduplication
    try:
        rows = similarity_search(query_vec, top_k=top_k * 3)
    except Exception as exc:
        logger.error("Database similarity search failed: %s", exc)
        raise RuntimeError(f"Retrieval failed: {exc}") from exc

    # 3. Deduplicate: keep only the highest-scoring result per unique chunk_text.
    # Rows are already sorted by descending score, so the first occurrence wins.
    seen: set[str] = set()
    unique_rows = []
    for row in rows:
        key = row["chunk_text"]
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
        if len(unique_rows) == top_k:
            break

    # 4. Map to Pydantic models
    results: List[RetrievalResult] = []
    for row in unique_rows:
        results.append(
            RetrievalResult(
                id=row["id"],
                book_name=row["book_name"],
                chapter=row.get("chapter", ""),
                page_number=row["page_number"],
                chunk_text=row["chunk_text"],
                similarity_score=round(row["similarity"], 4),
            )
        )

    if results:
        top_score = results[0].similarity_score
        logger.info(
            "Retrieved %d unique chunks (top similarity=%.3f).",
            len(results),
            top_score,
        )
    else:
        logger.warning("No chunks retrieved — database may be empty.")

    return results
