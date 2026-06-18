"""
Pydantic models for API requests, responses, and internal data structures.

These models provide validation, serialisation, and clear contracts
for every boundary in the application.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Request Models ───────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming question from the user."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The spiritual question to answer from the reference books.",
        examples=["Why do humans suffer?", "What is the nature of the self?"],
    )


# ── Response Models ──────────────────────────────────────────────────────


class Source(BaseModel):
    """A single source reference from the ingested books."""

    book_name: str = Field(..., description="Name of the source book.")
    chapter: str = Field(default="", description="Chapter name or number if detected.")
    excerpt: str = Field(
        ..., description="A short excerpt from the retrieved chunk."
    )
    similarity_score: float = Field(
        ..., description="Cosine similarity score (0-1)."
    )


class ChatResponse(BaseModel):
    """The chatbot's answer with supporting sources."""

    answer: str = Field(..., description="Generated answer from the teachings.")
    sources: List[Source] = Field(
        default_factory=list,
        description="List of source references used to generate the answer.",
    )
    question: str = Field(..., description="The original question (echoed).")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the response.",
    )


class IngestResponse(BaseModel):
    """Result of a document ingestion run."""

    status: str = Field(..., description="'success' or 'partial' or 'error'.")
    books_processed: int = Field(..., description="Number of PDFs processed.")
    total_chunks: int = Field(..., description="Total chunks created and stored.")
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during ingestion.",
    )
    elapsed_seconds: float = Field(
        ..., description="Total time taken for ingestion."
    )


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = Field(..., description="'healthy' or 'unhealthy'.")
    vector_store: str = Field(..., description="'connected' or 'disconnected'.")
    embedding_model: str = Field(..., description="Name of the loaded model.")
    chunk_count: int = Field(..., description="Total chunks in the vector store.")


# ── Internal Models ──────────────────────────────────────────────────────


class Chunk(BaseModel):
    """Internal representation of a text chunk with its embedding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    book_name: str
    page_number: int
    chunk_text: str
    embedding: Optional[List[float]] = None


class RetrievalResult(BaseModel):
    """A single result from vector similarity search.

    Note: *id* is a string because Qdrant uses UUIDs for point IDs.
    """

    id: str
    book_name: str
    chapter: str
    page_number: int
    chunk_text: str
    similarity_score: float


# ── Search-Only Models ───────────────────────────────────────────────────


class SearchResult(BaseModel):
    """A single chunk returned by the retrieval-only /search endpoint."""

    book_name: str = Field(..., description="Name of the source book.")
    page_number: int = Field(..., description="Page number in the book.")
    similarity: float = Field(..., description="Cosine similarity score (0-1).")
    chunk_text: str = Field(..., description="Full text of the retrieved chunk.")


class SearchResponse(BaseModel):
    """Response for the retrieval-only /search endpoint (no LLM call)."""

    question: str = Field(..., description="The original question (echoed).")
    results: List[SearchResult] = Field(
        default_factory=list,
        description="Top-k retrieved chunks ordered by descending similarity.",
    )
    total_indexed_chunks: int = Field(
        ..., description="Total number of chunks in the vector store."
    )
    message: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable message when no results pass the relevance threshold. "
            "Null when relevant results are found."
        ),
    )
