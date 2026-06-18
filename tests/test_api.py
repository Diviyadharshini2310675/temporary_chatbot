"""
Tests for the FastAPI endpoints.

Run with:  pytest tests/test_api.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI TestClient with patched dependencies."""
    # Patch the embedding model load and Qdrant init to avoid disk I/O
    with patch("app.main.get_model"), patch("app.main.init_collection"):
        from app.main import app

        with TestClient(app) as tc:
            yield tc


# ── Health Check ─────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for GET /health."""

    @patch("app.main.get_chunk_count", return_value=42)
    def test_health_healthy(self, mock_count, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["vector_store"] == "connected"
        assert data["chunk_count"] == 42
        assert "embedding_model" in data

    @patch("app.main.get_chunk_count", side_effect=Exception("Qdrant unavailable"))
    def test_health_vs_unavailable(self, mock_count, client):
        response = client.get("/health")
        assert response.status_code == 200  # Still returns 200, just reports unhealthy
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["vector_store"] == "disconnected"


# ── Chat Endpoint ───────────────────────────────────────────────────────


class TestChatEndpoint:
    """Tests for POST /chat."""

    @patch("app.main.retrieve_relevant_chunks")
    @patch("app.main.generate_answer")
    def test_chat_success(self, mock_generate, mock_retrieve, client):
        """Happy path: valid question → answer with sources."""
        from app.models import ChatResponse, RetrievalResult, Source

        mock_retrieve.return_value = [
            RetrievalResult(
                id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                book_name="Gita",
                page_number=23,
                chunk_text="The soul is eternal.",
                similarity_score=0.92,
            )
        ]
        mock_generate.return_value = ChatResponse(
            answer="According to the Gita, the soul is eternal.",
            sources=[
                Source(
                    book_name="Gita",
                    page_number=23,
                    excerpt="The soul is eternal.",
                    similarity_score=0.92,
                )
            ],
            question="What is the soul?",
        )

        response = client.post("/chat", json={"question": "What is the soul?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) == 1

    def test_chat_empty_question(self, client):
        """Empty question should return validation error."""
        response = client.post("/chat", json={"question": ""})
        assert response.status_code == 422  # Pydantic validation error

    def test_chat_missing_field(self, client):
        """Missing 'question' field should return validation error."""
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_question_too_long(self, client):
        """Question exceeding max length should be rejected."""
        response = client.post("/chat", json={"question": "x" * 2001})
        assert response.status_code == 422

    @patch("app.main.retrieve_relevant_chunks", return_value=[])
    def test_chat_no_chunks(self, mock_retrieve, client):
        """When no chunks exist, should return a helpful message."""
        response = client.post("/chat", json={"question": "What is the soul?"})
        assert response.status_code == 200
        data = response.json()
        assert "no reference materials" in data["answer"].lower()
        assert data["sources"] == []

    @patch("app.main.retrieve_relevant_chunks", side_effect=RuntimeError("Qdrant error"))
    def test_chat_retrieval_error(self, mock_retrieve, client):
        """Retrieval failures should return 500."""
        response = client.post("/chat", json={"question": "What is the soul?"})
        assert response.status_code == 500


# ── Ingest Endpoint ─────────────────────────────────────────────────────


class TestIngestEndpoint:
    """Tests for POST /ingest."""

    @patch("app.main.ingest_all_books")
    def test_ingest_success(self, mock_ingest, client):
        """Successful ingestion should return summary stats."""
        mock_ingest.return_value = {
            "status": "success",
            "books_processed": 3,
            "total_chunks": 150,
            "errors": [],
            "elapsed_seconds": 12.5,
        }

        response = client.post("/ingest")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["books_processed"] == 3
        assert data["total_chunks"] == 150

    @patch("app.main.ingest_all_books", side_effect=FileNotFoundError("No PDFs"))
    def test_ingest_no_pdfs(self, mock_ingest, client):
        """Missing PDFs should return 404."""
        response = client.post("/ingest")
        assert response.status_code == 404

    @patch("app.main.ingest_all_books", side_effect=Exception("Boom"))
    def test_ingest_unexpected_error(self, mock_ingest, client):
        """Unexpected errors should return 500."""
        response = client.post("/ingest")
        assert response.status_code == 500
