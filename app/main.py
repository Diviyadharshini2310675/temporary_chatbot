"""
FastAPI application entry-point for the Spiritual Assistant.

Provides:
  - POST /ingest  — Process all PDFs, create embeddings, store in Qdrant.
  - POST /chat    — Ask a question, get a RAG-powered answer with sources.
  - GET  /health  — Health-check endpoint.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.chatbot import generate_answer
from app.config import settings
from app.embeddings import get_model
from app.ingest import ingest_all_books
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    SearchResponse,
)
from app.qa_service import init_cache_table
from app.retrieve import retrieve_relevant_chunks
from app.vectordb import get_chunk_count, init_collection

# ── Logging Setup ────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("spiritual_assistant")


# ── Application Lifespan ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic for the FastAPI application.

    On startup:
      1. Initialise Qdrant collection.
      2. Pre-load the embedding model (avoids cold-start on first /chat).
    On shutdown:
      (Qdrant local storage handles persistence automatically.)
    """
    logger.info("=" * 60)
    logger.info("Spiritual Assistant starting up …")
    logger.info("=" * 60)

    # ── Qdrant collection initialisation ─────────────────────────────────
    try:
        init_collection()
        logger.info("Qdrant collection ready.")
    except Exception as exc:
        logger.critical("Qdrant init failed: %s", exc)

    # ── PostgreSQL cache table ────────────────────────────────────────────
    try:
        init_cache_table()
        logger.info("PostgreSQL chatbot_cache table ready.")
    except Exception as exc:
        logger.warning("chatbot_cache init failed (non-fatal): %s", exc)

    # ── Pre-load embedding model ─────────────────────────────────────────
    try:
        get_model()
        logger.info("Embedding model loaded and ready.")
    except Exception as exc:
        logger.critical("Embedding model load failed: %s", exc)

    logger.info("Startup complete — ready to serve requests.")
    yield  # ← Application runs here
    logger.info("Spiritual Assistant shutting down.")


# ── FastAPI App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Spiritual Assistant",
    description=(
        "A RAG-powered chatbot that answers spiritual questions "
        "using ONLY the teachings from uploaded reference books. "
        "Powered by sentence-transformers embeddings, Qdrant "
        "vector search, and OpenRouter for answer generation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/index.html")


# ── Endpoints ────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check the health of the application and its dependencies."""
    vs_status = "disconnected"
    model_name = settings.embedding_model
    count = 0

    try:
        count = get_chunk_count()
        vs_status = "connected"
    except Exception as exc:
        logger.warning("Health check — vector store: %s", exc)

    healthy = vs_status == "connected"

    return HealthResponse(
        status="healthy" if healthy else "unhealthy",
        vector_store=vs_status,
        embedding_model=model_name,
        chunk_count=count,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest() -> IngestResponse:
    """Ingest all PDFs from the books/ directory.

    Pipeline:
      1. Scan books/ for PDF files.
      2. Extract text page-by-page.
      3. Split into overlapping ~500-word chunks.
      4. Generate embeddings with all-MiniLM-L6-v2.
      5. Store chunks + embeddings in Qdrant (local embedded).
    """
    logger.info("POST /ingest — starting ingestion pipeline.")

    try:
        result = ingest_all_books()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion pipeline failed.")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return IngestResponse(**result)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Ask a spiritual question and get an answer from the reference books.

    The RAG pipeline:
      1. Encode the question → embedding vector.
      2. Retrieve top-5 most similar chunks via Qdrant cosine similarity.
      3. Build a strict prompt from the retrieved context.
      4. Generate an answer via OpenRouter (configurable model).
      5. Return the answer with source references.

    The model is **strictly instructed** to answer ONLY from the provided
    context.  If no relevant information is found, it will clearly state so
    rather than inventing an answer.
    """
    question = request.question.strip()
    logger.info("POST /chat — question: '%s'", question[:120])

    # ── Step 1+2: Retrieve relevant chunks ───────────────────────────────
    try:
        chunks = retrieve_relevant_chunks(question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not chunks:
        # No chunks in database at all — tell the user to ingest first.
        return ChatResponse(
            answer=(
                "No FAQ documents have been indexed yet. "
                "Please upload your FAQ PDFs to the books/ folder "
                "and run the ingestion process before asking questions."
            ),
            sources=[],
            question=question,
        )

    # ── Relevance threshold check ────────────────────────────────────────
    top_score = chunks[0].similarity_score
    threshold = settings.similarity_threshold
    logger.info(
        "Top score: %.4f | Threshold: %.2f | %s",
        top_score,
        threshold,
        "PASSED" if top_score >= threshold else "FAILED",
    )

    if top_score < threshold:
        logger.warning(
            "Question below relevance threshold — returning not-found response."
        )
        return ChatResponse(
            answer=(
                "This information was not found in the uploaded FAQ documents."
            ),
            sources=[],
            question=question,
        )

    # ── Step 3+4: Generate answer ────────────────────────────────────────
    try:
        response = generate_answer(question, chunks)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return response


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: ChatRequest) -> SearchResponse:
    """Retrieval-only endpoint — returns top chunks WITHOUT calling OpenRouter.

    Use this to verify retrieval quality before enabling answer generation.

    Pipeline:
      1. Encode the question → embedding vector.
      2. Retrieve top-5 most similar chunks via Qdrant cosine similarity.
      3. Return chunks with full text and similarity scores.

    No API key required (embedding model runs locally).
    """
    from app.models import SearchResult

    question = request.question.strip()
    logger.info("POST /search — question: '%s'", question[:120])

    # ── Retrieve relevant chunks ───────────────────────────────────────────
    try:
        chunks = retrieve_relevant_chunks(question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # ── Total chunk count for context ──────────────────────────────────────
    try:
        total = get_chunk_count()
    except Exception:
        total = 0

    # ── Relevance threshold check ──────────────────────────────────────────
    threshold = settings.similarity_threshold
    if chunks:
        top_score = chunks[0].similarity_score
    else:
        top_score = 0.0

    logger.info(
        "Top score: %.4f | Threshold: %.2f | %s",
        top_score,
        threshold,
        "PASSED" if top_score >= threshold else "FAILED",
    )

    if top_score < threshold:
        logger.warning(
            "Search below relevance threshold — returning empty results."
        )
        return SearchResponse(
            question=question,
            results=[],
            total_indexed_chunks=total,
            message=(
                "No sufficiently relevant information found "
                "in the reference books."
            ),
        )

    # ── Build results ──────────────────────────────────────────────────────
    results: list[SearchResult] = []
    for chunk in chunks:
        results.append(
            SearchResult(
                book_name=chunk.book_name,
                page_number=chunk.page_number,
                similarity=chunk.similarity_score,
                chunk_text=chunk.chunk_text,
            )
        )

    return SearchResponse(
        question=question,
        results=results,
        total_indexed_chunks=total,
    )


# ── Direct Execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
