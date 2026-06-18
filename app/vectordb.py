"""
Qdrant vector database layer (local embedded mode).

Handles collection creation, point insertion, deletion, and
vector similarity search.  Uses on-disk storage — no server needed.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings

logger = logging.getLogger(__name__)

# ── Module-level client (lazy singleton) ─────────────────────────────────
_client: Optional[QdrantClient] = None
_collection_ready: bool = False


def get_client() -> QdrantClient:
    """Return the cached QdrantClient, creating it on first call.

    Uses local embedded storage at the path configured in settings.
    No external server process is required.
    """
    global _client
    if _client is None:
        path = settings.qdrant_abs_path
        logger.info("Opening Qdrant local storage at '%s' …", path)
        _client = QdrantClient(path=path)
    return _client


def init_collection() -> None:
    """Create the 'spiritual_books' collection if it does not exist.

    The collection uses cosine distance and 384-dimensional vectors
    (matching all-MiniLM-L6-v2 output).

    Idempotent — safe to call on every startup.
    """
    global _collection_ready
    if _collection_ready:
        return

    client = get_client()
    collection_name = settings.qdrant_collection
    dim = settings.embedding_dimension

    # Check whether the collection already exists.
    collections = [
        c.name for c in client.get_collections().collections  # type: ignore[union-attr]
    ]

    if collection_name not in collections:
        logger.info(
            "Creating collection '%s' (dim=%d, distance=Cosine) …",
            collection_name,
            dim,
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    else:
        logger.info("Collection '%s' already exists.", collection_name)

    _collection_ready = True


# ── CRUD Operations ──────────────────────────────────────────────────────


def clear_collection() -> None:
    """Delete and recreate the entire Qdrant collection.

    Removes **all** points and payloads, then recreates the collection
    with the same vector configuration.  This guarantees a clean slate
    for re-ingestion — old book1/book2 entries are permanently removed.

    Idempotent — safe to call on every ingestion run.
    """
    client = get_client()
    collection_name = settings.qdrant_collection
    dim = settings.embedding_dimension

    # Drop the collection if it already exists
    collections = [
        c.name for c in client.get_collections().collections  # type: ignore[union-attr]
    ]
    if collection_name in collections:
        logger.info("Deleting existing collection '%s' …", collection_name)
        client.delete_collection(collection_name=collection_name)
        logger.info("Collection '%s' deleted.", collection_name)

    # Recreate with the same schema
    logger.info(
        "Creating collection '%s' (dim=%d, distance=Cosine) …",
        collection_name,
        dim,
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("Collection '%s' recreated.", collection_name)

    global _collection_ready
    _collection_ready = True


def clear_book(book_name: str) -> int:
    """Delete all points belonging to *book_name* from the collection.

    Returns an estimate of deleted points (Qdrant reports the operation
    as acknowledged; the actual count is returned from a pre-delete scroll).
    Useful before re-ingesting the same book (idempotent ingestion).
    """
    client = get_client()
    collection_name = settings.qdrant_collection

    # Count points for this book before deleting
    count_result = client.count(
        collection_name=collection_name,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="book_name",
                    match=MatchValue(value=book_name),
                )
            ]
        ),
    )
    before_count: int = count_result.count  # type: ignore[assignment]

    if before_count == 0:
        return 0

    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="book_name",
                    match=MatchValue(value=book_name),
                )
            ]
        ),
    )
    logger.info("Cleared %d existing points for '%s'.", before_count, book_name)
    return before_count


def insert_chunks(
    chunks: List[tuple[str, int, str, str, List[float]]],
) -> int:
    """Batch-insert chunks with their embeddings into Qdrant.

    Args:
        chunks: List of (book_name, page_number, chunk_text, chapter, embedding).

    Returns:
        Number of points upserted.
    """
    if not chunks:
        return 0

    client = get_client()
    collection_name = settings.qdrant_collection

    points: List[PointStruct] = []
    for book_name, page_number, chunk_text, chapter, embedding in chunks:
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "book_name": book_name,
                    "page_number": page_number,
                    "chunk_text": chunk_text,
                    "chapter": chapter,
                },
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )
    logger.info("Upserted %d points into '%s'.", len(points), collection_name)
    return len(points)


def get_chunk_count() -> int:
    """Return the total number of points in the collection."""
    client = get_client()
    count_result = client.count(
        collection_name=settings.qdrant_collection,
        exact=True,
    )
    return count_result.count  # type: ignore[return-value]


def get_book_list() -> List[str]:
    """Return a sorted list of distinct book names in the collection.

    Uses a scroll with value-based payload indexing to collect unique names.
    """
    client = get_client()
    books: set[str] = set()

    records, next_offset = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=100,
        with_payload=["book_name"],
        with_vectors=False,
    )

    for record in records:
        if record.payload and "book_name" in record.payload:
            books.add(record.payload["book_name"])  # type: ignore[index]

    # Continue scrolling if there are more points
    while next_offset is not None:
        records, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            offset=next_offset,
            limit=100,
            with_payload=["book_name"],
            with_vectors=False,
        )
        for record in records:
            if record.payload and "book_name" in record.payload:
                books.add(record.payload["book_name"])  # type: ignore[index]

    return sorted(books)


# ── Vector Similarity Search ─────────────────────────────────────────────


def similarity_search(
    query_embedding: List[float],
    top_k: int | None = None,
) -> List[dict]:
    """Return the *top_k* most similar chunks by cosine similarity.

    Uses ``query_points`` (the local-mode search API) with a raw
    vector query.  Qdrant's cosine distance is converted to a
    similarity score: ``similarity = 1.0 - distance``.

    Args:
        query_embedding: The embedding vector of the user's question.
        top_k: Number of results (defaults to settings.top_k_results).

    Returns:
        List of dicts with keys:
            id (str), book_name, page_number, chunk_text, similarity (float).
    """
    if top_k is None:
        top_k = settings.top_k_results

    client = get_client()

    # query_points is the unified search API in qdrant-client ≥ 1.12.
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    output: List[dict] = []
    for point in response.points:  # type: ignore[union-attr]
        payload = point.payload or {}
        output.append(
            {
                "id": str(point.id),
                "book_name": payload.get("book_name", "Unknown"),
                "page_number": payload.get("page_number", 0),
                "chunk_text": payload.get("chunk_text", ""),
                "chapter": payload.get("chapter", ""),
                # Qdrant with Distance.COSINE returns cosine similarity
                # as the score (higher = more similar).  Clamp to [0, 1]
                # because the raw dot product can dip slightly below 0
                # for unrelated vectors.
                "similarity": round(max(0.0, min(1.0, point.score)), 4),
            }
        )

    logger.info(
        "Similarity search returned %d results (top_k=%d).",
        len(output),
        top_k,
    )
    return output
