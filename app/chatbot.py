"""
Answer generation using OpenRouter's chat API with RAG context.

Builds a strict prompt from retrieved document chunks and instructs
the model to answer ONLY from the provided context.  If the answer
cannot be found, the model must clearly state so rather than invent.

Uses the OpenAI-compatible SDK pointed at the OpenRouter base URL.
Any model available on OpenRouter can be configured via .env
(OPENROUTER_MODEL).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from openai import OpenAI

from app.config import settings
from app.models import ChatResponse, RetrievalResult, Source
from app.qa_service import find_qa_answer, get_cached_response, save_cached_response, save_user_question

logger = logging.getLogger(__name__)

# ── Module-level OpenRouter client ─────────────────────────────────────────
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return the cached OpenAI client pointed at OpenRouter, creating it on first call.

    OpenRouter exposes an OpenAI-compatible endpoint so we use the
    standard ``openai`` SDK with a custom ``base_url``.
    """
    global _client
    if _client is None:
        if (
            not settings.openrouter_api_key
            or "your-" in settings.openrouter_api_key
        ):
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured. "
                "Set it in your .env file."
            )
        logger.info(
            "Creating OpenRouter client (base_url=%s) …",
            settings.openrouter_base_url,
        )
        _client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
    return _client


# ── System Prompt ────────────────────────────────────────────────────────
# This is the core behavioural contract for the LLM.  It is designed to
# prevent hallucination and enforce source-only answers.

SYSTEM_PROMPT = """You are a compassionate spiritual assistant.

Your job is to answer questions using the reference passages provided by the user.

Rules:
- Base your answer on the themes, teachings, and ideas present in the passages.
- The question may be phrased differently from the passage text — use your understanding to connect the two.
- Do NOT add teachings, quotes, or ideas that are not present in the passages.
- If the passages are genuinely unrelated to the question, reply: 'This information was not found in the uploaded FAQ documents.'
- Write a warm, clear, helpful answer in natural language.
- Do NOT add book names, page numbers, or citations at the end of your answer. Sources are shown separately by the interface."""


def _build_user_prompt(
    question: str,
    context_chunks: List[RetrievalResult],
) -> str:
    """Assemble the user prompt with context passages and the question.

    The structure:
        1. Reference passages (numbered)
        2. The user's question
        3. Explicit instruction to answer from passages only
    """
    if not context_chunks:
        return f"""No reference passages were found for this question.

The user asks: {question}

Since no relevant context was retrieved, you must tell the user:
'This information was not found in the uploaded FAQ documents.'"""

    # Build the context section
    passages: List[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        passages.append(
            f"[{i}] (From **{chunk.book_name}**, Page {chunk.page_number})\n"
            f"    {chunk.chunk_text}"
        )

    context_block = "\n\n".join(passages)

    return f"""## Reference Passages

Below are passages from the FAQ documents that may contain relevant information.

{context_block}

## Instructions

The passages above are from a spiritual library. Using the ideas and teachings in these passages,
answer the question below. The question may use different wording than the passages — look for
the underlying meaning and relevant teachings. Only use what is in the passages.
Do NOT append book names or page numbers at the end — sources are displayed separately.

## Question

{question}"""


def generate_answer(
    question: str,
    retrieved_chunks: List[RetrievalResult],
) -> ChatResponse:
    """Generate an answer from retrieved context via OpenRouter.

    Args:
        question: The user's original question.
        retrieved_chunks: Top-k chunks from vector similarity search.

    Returns:
        ChatResponse with answer, sources, question echo, and timestamp.

    Raises:
        RuntimeError: If the OpenRouter API call fails.
        ValueError: If the question is empty.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    # ── Step 0: Check chatbot_cache (fastest — no Qdrant, no LLM) ────────
    cached = get_cached_response(question)
    if cached:
        sources = [Source(**s) for s in cached["sources"]]
        return ChatResponse(
            answer=cached["answer"],
            sources=sources,
            question=question.strip(),
            timestamp=datetime.now(timezone.utc),
        )

    # ── Step 1: Save question + check PostgreSQL Q&A dataset ─────────────
    save_user_question(question)

    cached_answer = find_qa_answer(question)
    if cached_answer:
        logger.info("Returning cached answer from qa_dataset.")
        return ChatResponse(
            answer=cached_answer,
            sources=[],
            question=question.strip(),
            timestamp=datetime.now(timezone.utc),
        )

    client = _get_client()
    user_prompt = _build_user_prompt(question, retrieved_chunks)

    logger.info(
        "OpenRouter request started (model=%s, context_chunks=%d) …",
        settings.openrouter_model,
        len(retrieved_chunks),
    )

    try:
        response = client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # Low temperature for faithfulness
            max_tokens=800,
        )
    except Exception as exc:
        logger.error("OpenRouter API call failed: %s", exc)
        raise RuntimeError(
            f"Answer generation failed via OpenRouter: {exc}"
        ) from exc

    answer_text = response.choices[0].message.content or ""
    logger.info(
        "OpenRouter response received (%d chars, model=%s).",
        len(answer_text),
        settings.openrouter_model,
    )

    # ── Build source list ────────────────────────────────────────────────
    sources: List[Source] = []
    for chunk in retrieved_chunks:
        excerpt = chunk.chunk_text[:200]
        if len(chunk.chunk_text) > 200:
            excerpt += "…"
        sources.append(
            Source(
                book_name=chunk.book_name,
                chapter=chunk.chapter,
                excerpt=excerpt,
                similarity_score=chunk.similarity_score,
            )
        )

    # ── Step 5: Save to chatbot_cache for future requests ────────────────
    sources_as_dicts = [s.model_dump() for s in sources]
    save_cached_response(question, answer_text.strip(), sources_as_dicts)

    return ChatResponse(
        answer=answer_text.strip(),
        sources=sources,
        question=question.strip(),
        timestamp=datetime.now(timezone.utc),
    )
