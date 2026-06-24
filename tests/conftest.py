"""
Shared fixtures for the Spiritual Assistant test suite.

All fixtures use in-memory / mock equivalents so tests run without
external dependencies (no Qdrant, no OpenAI key required).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_settings():
    """Override all external-service settings for every test."""
    with patch.dict(
        os.environ,
        {
            "QDRANT_PATH": "./qdrant_data",
            "QDRANT_COLLECTION": "spiritual_books",
            "OPENROUTER_API_KEY": "sk-or-test-mock-key-12345",
            "OPENROUTER_MODEL": "google/gemini-2.5-flash",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
            "EMBEDDING_DIMENSION": "384",
            "CHUNK_SIZE_WORDS": "500",
            "CHUNK_OVERLAP_WORDS": "100",
            "TOP_K_RESULTS": "2",
            "SIMILARITY_THRESHOLD": "0.35",
            "LOG_LEVEL": "WARNING",
            # PostgreSQL — point at a non-existent host so no accidental live calls
            "PG_HOST": "localhost",
            "PG_PORT": "5432",
            "PG_DBNAME": "spiritual_chatbot_test",
            "PG_USER": "postgres",
            "PG_PASSWORD": "",
        },
        clear=False,
    ):
        yield


@pytest.fixture(autouse=True)
def mock_qa_service():
    """Stub out PostgreSQL Q&A calls for all tests.

    - save_user_question: no-op (returns None)
    - find_qa_answer: returns None by default (RAG path is taken)

    Individual tests that need a cache-hit can override find_qa_answer
    via a nested patch.
    """
    with (
        patch("app.chatbot.save_user_question", return_value=None),
        patch("app.chatbot.find_qa_answer", return_value=None),
    ):
        yield


@pytest.fixture
def sample_chunks():
    """Return a list of mock RetrievalResult objects (Qdrant string IDs)."""
    from app.models import RetrievalResult

    return [
        RetrievalResult(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            book_name="The Bhagavad Gita",
            page_number=23,
            chunk_text=(
                "The soul is neither born, nor does it ever die. "
                "Having once existed, it never ceases to be. "
                "It is unborn, eternal, ever-existing and primeval."
            ),
            similarity_score=0.92,
        ),
        RetrievalResult(
            id="b2c3d4e5-f6a7-8901-bcde-f12345678901",
            book_name="The Bhagavad Gita",
            page_number=47,
            chunk_text=(
                "One who sees inaction in action, and action in inaction, "
                "is wise among men. Such a person is a yogi, having "
                "performed all actions."
            ),
            similarity_score=0.85,
        ),
        RetrievalResult(
            id="c3d4e5f6-a7b8-9012-cdef-123456789012",
            book_name="The Dhammapada",
            page_number=12,
            chunk_text=(
                "All that we are is the result of what we have thought. "
                "It is founded on our thoughts. It is made up of our thoughts."
            ),
            similarity_score=0.78,
        ),
    ]


@pytest.fixture
def sample_chunks_dict():
    """Return mock retrieval results as raw dicts (matching Qdrant output)."""
    return [
        {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "book_name": "The Bhagavad Gita",
            "page_number": 23,
            "chunk_text": "The soul is neither born, nor does it ever die.",
            "similarity": 0.92,
        },
        {
            "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
            "book_name": "The Bhagavad Gita",
            "page_number": 47,
            "chunk_text": "One who sees inaction in action, and action in inaction.",
            "similarity": 0.85,
        },
    ]


@pytest.fixture
def mock_openai_response():
    """Return a mock OpenAI chat completion response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = (
        "According to the Bhagavad Gita, the soul is eternal and never dies. "
        "It is unborn, ever-existing, and primeval. "
        "[Source: The Bhagavad Gita, Page 23]"
    )
    return mock
