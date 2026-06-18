"""
Tests for the chatbot answer generation module.

Run with:  pytest tests/test_chatbot.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.chatbot import _build_user_prompt, SYSTEM_PROMPT, generate_answer
from app.models import ChatResponse, RetrievalResult


# ── Prompt Building ──────────────────────────────────────────────────────


class TestBuildUserPrompt:
    """Tests for the prompt assembly logic."""

    def test_prompt_includes_all_chunks(self, sample_chunks):
        prompt = _build_user_prompt("What is the soul?", sample_chunks)
        for chunk in sample_chunks:
            assert chunk.book_name in prompt
            assert chunk.chunk_text[:50] in prompt
            assert str(chunk.page_number) in prompt

    def test_prompt_includes_question(self, sample_chunks):
        prompt = _build_user_prompt("Why do humans suffer?", sample_chunks)
        assert "Why do humans suffer?" in prompt

    def test_prompt_includes_instructions(self, sample_chunks):
        prompt = _build_user_prompt("What is karma?", sample_chunks)
        assert "Reference Passages" in prompt
        assert "only" in prompt.lower()

    def test_empty_context_returns_guidance(self):
        prompt = _build_user_prompt("What is the meaning of life?", [])
        assert "No reference passages were found" in prompt
        assert "What is the meaning of life?" in prompt


# ── System Prompt ────────────────────────────────────────────────────────


class TestSystemPrompt:
    """Verify the system prompt contract."""

    def test_forbids_invention(self):
        assert "Do NOT invent" in SYSTEM_PROMPT

    def test_requires_sources(self):
        assert "source citations" in SYSTEM_PROMPT

    def test_defines_not_found_behaviour(self):
        assert "I could not find this information" in SYSTEM_PROMPT

    def test_answer_only_from_passages(self):
        lower = SYSTEM_PROMPT.lower()
        assert "answer only" in lower and "supplied reference context" in lower


# ── Answer Generation ────────────────────────────────────────────────────


class TestGenerateAnswer:
    """Tests for the OpenRouter-powered answer generation."""

    @patch("app.chatbot.settings.openrouter_api_key", "sk-test-mock-key")
    @patch("app.chatbot.OpenAI")
    def test_generates_answer_with_sources(
        self, mock_openai_cls, sample_chunks, mock_openai_response
    ):
        """Happy path: chunks + valid OpenAI response → ChatResponse with sources."""
        # Reset the module-level client so our mock is used
        import app.chatbot as cb
        cb._client = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        result = generate_answer("What is the soul?", sample_chunks)

        assert isinstance(result, ChatResponse)
        assert len(result.answer) > 0
        assert "Bhagavad Gita" in result.answer
        assert len(result.sources) == len(sample_chunks)
        assert result.sources[0].book_name == "The Bhagavad Gita"
        assert result.sources[0].page_number == 23
        assert result.question == "What is the soul?"

    @patch("app.chatbot.settings.openrouter_api_key", "sk-test-mock-key")
    @patch("app.chatbot.OpenAI")
    def test_empty_question_raises(self, mock_openai_cls):
        """Empty question should raise ValueError."""
        with pytest.raises(ValueError, match="Question must not be empty"):
            generate_answer("", [])

    @patch("app.chatbot.settings.openrouter_api_key", "sk-test-mock-key")
    @patch("app.chatbot.OpenAI")
    def test_whitespace_question_raises(self, mock_openai_cls):
        """Whitespace-only question should raise ValueError."""
        with pytest.raises(ValueError, match="Question must not be empty"):
            generate_answer("   \n  ", [])

    @patch("app.chatbot.settings.openrouter_api_key", "sk-test-mock-key")
    @patch("app.chatbot.OpenAI")
    def test_no_context_handles_gracefully(
        self, mock_openai_cls, mock_openai_response
    ):
        """When no chunks are retrieved, the answer should still be generated."""
        import app.chatbot as cb
        cb._client = None

        # Override the response for the "no context" case
        mock_openai_response.choices[0].message.content = (
            "I could not find an answer to this question in the reference material."
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        result = generate_answer("What is the soul?", [])
        assert isinstance(result, ChatResponse)
        assert len(result.sources) == 0
        assert "not find" in result.answer.lower() or "no reference" in result.answer.lower()

    @patch("app.chatbot.settings.openrouter_api_key", "sk-test-mock-key")
    @patch("app.chatbot.OpenAI")
    def test_sources_have_excerpts(self, mock_openai_cls, sample_chunks, mock_openai_response):
        """Each source should include an excerpt from the chunk."""
        import app.chatbot as cb
        cb._client = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        result = generate_answer("What is the soul?", sample_chunks)

        for source in result.sources:
            assert len(source.excerpt) > 0
            assert source.similarity_score > 0

    @patch("app.chatbot.settings.openrouter_api_key", "sk-test-mock-key")
    @patch("app.chatbot.OpenAI")
    def test_openrouter_error_propagates(self, mock_openai_cls, sample_chunks):
        """OpenRouter API failures should raise RuntimeError."""
        import app.chatbot as cb
        cb._client = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API rate limit")
        mock_openai_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="Answer generation failed"):
            generate_answer("What is karma?", sample_chunks)


# ── Response Model Validation ────────────────────────────────────────────


class TestChatResponseModel:
    """Verify the Pydantic response model behaves correctly."""

    def test_valid_response(self):
        from app.models import Source

        response = ChatResponse(
            answer="The soul is eternal.",
            sources=[
                Source(
                    book_name="Gita",
                    page_number=23,
                    excerpt="The soul...",
                    similarity_score=0.92,
                )
            ],
            question="What is the soul?",
        )
        assert response.answer == "The soul is eternal."
        assert len(response.sources) == 1
        assert response.timestamp is not None
