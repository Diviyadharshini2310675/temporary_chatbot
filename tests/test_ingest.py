"""
Tests for the PDF ingestion and chunking pipeline.

Run with:  pytest tests/test_ingest.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingest import _chunk_pages, _normalise_text
from app.config import settings


# ── Text Normalisation ───────────────────────────────────────────────────


class TestNormaliseText:
    """Tests for _normalise_text helper."""

    def test_collapses_multiple_spaces(self):
        result = _normalise_text("hello    world")
        assert result == "hello world"

    def test_collapses_newlines(self):
        result = _normalise_text("line one\nline two\n\nline three")
        assert result == "line one line two line three"

    def test_rejoins_hyphenated_words(self):
        result = _normalise_text("teach-\nings about life")
        assert result == "teachings about life"

    def test_strips_whitespace(self):
        result = _normalise_text("  \n  hello world  \n  ")
        assert result == "hello world"

    def test_empty_string(self):
        result = _normalise_text("   \n\n  ")
        assert result == ""


# ── Chunking ─────────────────────────────────────────────────────────────


class TestChunkPages:
    """Tests for _chunk_pages chunking logic."""

    def test_single_short_page(self):
        """A page shorter than chunk_size → one chunk."""
        pages = [(1, "Only twenty words in this page to test chunking behaviour")]
        chunks = _chunk_pages("Test Book", pages)
        assert len(chunks) == 1
        assert chunks[0][0] == "Test Book"
        assert chunks[0][1] == 1

    def test_page_longer_than_chunk_size(self):
        """A page longer than chunk_size → multiple overlapping chunks."""
        words = ["spiritual"] * 600
        pages = [(1, " ".join(words))]
        chunks = _chunk_pages("Test Book", pages)
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
        for chunk in chunks:
            assert chunk[0] == "Test Book"
            assert chunk[1] == 1

    def test_overlap_preserved(self):
        """Overlapping chunks should share some words."""
        words = [f"word{i}" for i in range(600)]
        pages = [(1, " ".join(words))]
        chunks = _chunk_pages("Test Book", pages)
        chunk1_words = set(chunks[0][2].split())
        chunk2_words = set(chunks[1][2].split())
        overlap = chunk1_words & chunk2_words
        assert len(overlap) > 0, "Chunks should have overlapping words"

    def test_multiple_pages(self):
        """Chunks across multiple pages preserve page numbers."""
        pages = [
            (1, "first page content " * 30),
            (2, "second page content " * 30),
            (3, "third page content " * 30),
        ]
        chunks = _chunk_pages("Multi Page", pages)
        assert len(chunks) == 3
        assert chunks[0][1] == 1
        assert chunks[1][1] == 2
        assert chunks[2][1] == 3

    def test_empty_pages_produces_no_chunks(self):
        chunks = _chunk_pages("Empty", [])
        assert chunks == []

    def test_page_with_few_words(self):
        """A page with just a few words should still produce a chunk."""
        pages = [(1, "Om shanti")]
        chunks = _chunk_pages("Short Book", pages)
        assert len(chunks) == 1


# ── Full Ingest Pipeline (with mocks) ────────────────────────────────────


class TestIngestAllBooks:
    """Integration-style tests for the full ingestion pipeline with mocked deps."""

    @patch("app.ingest.encode_chunks")
    @patch("app.ingest.insert_chunks")
    @patch("app.ingest.clear_book")
    @patch("app.ingest.clear_collection")
    @patch("app.ingest.PdfReader")
    def test_ingest_success(
        self,
        mock_reader_cls,
        mock_clear_collection,
        mock_clear,
        mock_insert,
        mock_encode,
        tmp_path,
    ):
        """Full ingest pipeline should run without errors."""
        from app.ingest import ingest_all_books

        # Setup: create a mock PDF in a temp books dir
        books_dir = tmp_path / "books"
        books_dir.mkdir()
        pdf_path = books_dir / "Test Book.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 mock content")

        # Mock PDF reader to return one page of text
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "The soul is eternal and never dies."
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        # Mock embedding: return 384-dim vector
        mock_encode.return_value = [[0.1] * 384]
        mock_insert.return_value = 1
        mock_clear.return_value = 0

        # Override books_dir for this test
        with patch.object(settings, "books_dir", books_dir):
            result = ingest_all_books()

        assert result["status"] == "success"
        assert result["books_processed"] == 1
        assert result["total_chunks"] == 1

    @patch("app.ingest.encode_chunks")
    @patch("app.ingest.insert_chunks")
    @patch("app.ingest.clear_book")
    def test_no_pdfs_raises(self, mock_clear, mock_insert, mock_encode, tmp_path):
        """When no PDFs exist in the books dir, should raise FileNotFoundError."""
        from app.ingest import ingest_all_books

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch.object(settings, "books_dir", empty_dir):
            with pytest.raises(FileNotFoundError, match="No PDF"):
                ingest_all_books()
