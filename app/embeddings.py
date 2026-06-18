"""
Embedding generation using sentence-transformers.

Wraps the local embedding model (all-MiniLM-L6-v2) and provides
a simple interface for encoding both individual queries and batches
of text chunks.
"""

from __future__ import annotations

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

# ── Module-level model instance ──────────────────────────────────────────
# Loaded once at import time and reused — the model is ~90 MB and
# loading it per-request would be prohibitively slow.
_embedding_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the cached SentenceTransformer model, loading it on first call.

    The model is stored in the project's embeddings/ directory for
    persistence across environment rebuilds.
    """
    global _embedding_model
    if _embedding_model is None:
        model_name = settings.embedding_model
        cache_dir = str(settings.embeddings_cache_dir)
        logger.info("Loading embedding model '%s' …", model_name)
        try:
            _embedding_model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
            )
            logger.info(
                "Embedding model loaded (dim=%d).",
                _embedding_model.get_embedding_dimension(),
            )
        except Exception as exc:
            logger.critical("Failed to load embedding model: %s", exc)
            raise RuntimeError(
                f"Embedding model load failed: {exc}"
            ) from exc
    return _embedding_model


def encode_query(question: str) -> List[float]:
    """Encode a single user question into an embedding vector.

    Args:
        question: The raw text of the user's question.

    Returns:
        A list of floats (length = embedding_dimension, typically 384).
    """
    model = get_model()
    embedding = model.encode(question, normalize_embeddings=True)
    return embedding.tolist()


def encode_chunks(texts: List[str], show_progress: bool = True) -> List[List[float]]:
    """Encode a batch of text chunks into embedding vectors.

    Args:
        texts: List of chunk strings.
        show_progress: Whether to display a progress bar via tqdm.

    Returns:
        List of embedding vectors, each a list of floats.
    """
    if not texts:
        return []

    model = get_model()
    logger.info("Encoding %d chunks …", len(texts))
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        batch_size=32,
    )
    logger.info("Encoding complete (%d vectors).", len(embeddings))
    return [e.tolist() for e in embeddings]
