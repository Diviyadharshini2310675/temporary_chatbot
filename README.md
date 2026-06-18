# Spiritual Assistant 🕉️

A production-ready **RAG (Retrieval-Augmented Generation) chatbot** that answers spiritual questions **only** from uploaded reference books.  Built with FastAPI, Qdrant (local embedded), sentence-transformers, and OpenAI.

---

## Architecture

```
                    ┌──────────────┐
                    │   User       │
                    │  "Why do we  │
                    │   suffer?"   │
                    └──────┬───────┘
                           │ POST /chat
                           ▼
┌──────────────────────────────────────────────────┐
│                  FastAPI Server                    │
│                                                    │
│  1. /chat  ──► encode query ──► Qdrant search    │
│                                    │               │
│  2. /ingest ──► PDF extraction ──► chunking       │
│                 ──► embeddings   ──► store in DB  │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────┐ │
│  │ sentence │  │  Qdrant  │  │  OpenAI GPT-4o  │ │
│  │transform │  │ (cosine  │  │  mini (prompt   │ │
│  │  ers     │  │  sim)    │  │   + context)    │ │
│  └──────────┘  └──────────┘  └─────────────────┘ │
└──────────────────────────────────────────────────┘
```

**Safety**: The LLM is strictly prompted to answer ONLY from the provided context.  If no relevant passages are found, it clearly states so — it never invents teachings.

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- An **OpenAI API key**
- No external database required — Qdrant runs as an embedded local store

### 1. Clone & Install

```bash
cd spiritual-chatbot
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` and set:
- `OPENAI_API_KEY` — your OpenAI API key
- `QDRANT_PATH` — where to store vector data (default: `./qdrant_data`)

### 3. Add Your Books

Place your spiritual PDF books in the `books/` directory:

```
books/
├── The Bhagavad Gita.pdf
├── The Dhammapada.pdf
├── The Upanishads.pdf
├── Tao Te Ching.pdf
└── The Holy Bible.pdf
```

### 4. Ingest Books

```bash
curl -X POST http://localhost:8000/ingest
```

This will:
1. Read all PDFs from `books/`
2. Extract text page-by-page
3. Split into ~500-word overlapping chunks
4. Generate 384-dimension embeddings with `all-MiniLM-L6-v2`
5. Store everything in PostgreSQL with pgvector

### 5. Ask Questions

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Why do humans suffer?"}'
```

**Response:**

```json
{
  "answer": "According to the Bhagavad Gita, suffering arises from attachment...",
  "sources": [
    {
      "book_name": "The Bhagavad Gita",
      "page_number": 23,
      "excerpt": "From attachment springs desire...",
      "similarity_score": 0.92
    }
  ],
  "question": "Why do humans suffer?",
  "timestamp": "2026-06-05T10:30:00Z"
}
```

### 6. Start the Server

```bash
python -m app.main
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Reference

| Method | Path      | Description                                              |
|--------|-----------|----------------------------------------------------------|
| `POST` | `/ingest` | Process all PDFs, create embeddings, store in database   |
| `POST` | `/chat`   | Ask a question → get RAG-powered answer with sources     |
| `GET`  | `/health` | Health check (database, embedding model, chunk count)    |

### POST /chat

**Request:**
```json
{
  "question": "What is the nature of the self?"
}
```

**Success (200):**
```json
{
  "answer": "...",
  "sources": [
    {
      "book_name": "string",
      "page_number": 0,
      "excerpt": "string",
      "similarity_score": 0.0
    }
  ],
  "question": "string",
  "timestamp": "2026-06-05T00:00:00Z"
}
```

**Error Codes:**
- `400` — Invalid question (empty)
- `422` — Validation error (missing field, too long)
- `500` — Server error (database down, API failure)

---

## Project Structure

```
spiritual-chatbot/
│
├── books/                     # ← Place your PDFs here
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI application & endpoints
│   ├── config.py             # Settings from .env
│   ├── vectordb.py           # Qdrant vector database layer
│   ├── models.py             # Pydantic request/response models
│   ├── embeddings.py         # SentenceTransformer wrapper
│   ├── ingest.py             # PDF extraction & chunking pipeline
│   ├── retrieve.py           # Vector similarity search
│   └── chatbot.py            # OpenAI answer generation (RAG)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures & mocks
│   ├── test_ingest.py        # PDF extraction & chunking tests
│   ├── test_chatbot.py       # Prompt building & answer generation tests
│   └── test_api.py           # HTTP endpoint tests
│
├── embeddings/               # Cached embedding model files
├── .env                      # Secrets & configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Key Design Decisions

### Why all-MiniLM-L6-v2?

- **384 dimensions** — compact, fast similarity search
- **Runs locally** — no API cost for embeddings
- **Excellent quality/speed tradeoff** — 80%+ of larger model quality at a fraction of the cost
- **~90 MB on disk** — caches to `embeddings/`

### Why Qdrant (Local Embedded)?

- **Zero external dependencies** — no server process, no Docker, runs as an embedded library
- **On-disk storage** — data persists to `qdrant_data/`, survives restarts
- **Cosine similarity** — native support, fast exact search on up to ~1M vectors
- **Filtered search** — delete/replace by book name for idempotent ingestion
- **Production-ready** — same API as Qdrant Cloud/server mode, easy to scale later

### Chunking Strategy

- **500 words per chunk** — balances context richness with retrieval precision
- **100-word overlap** — prevents splitting key concepts across chunk boundaries
- **Page-level granularity** — preserves source traceability

### Safety & Faithfulness

- **Temperature = 0.3** — low randomness, high faithfulness to context
- **Strict system prompt** — the model is explicitly instructed to refuse answering from outside knowledge
- **Source citations** — every answer includes book name and page number
- **Explicit "not found" behavior** — model states clearly when information is unavailable

---

## Scaling to 100+ Books

The system is designed to scale:

1. **Bulk ingestion** — batch embedding and upsert for fast processing
2. **Local Qdrant** — handles up to ~1M vectors with exact cosine search; switch to Qdrant Cloud for billions
3. **Idempotent ingestion** — re-running clears and replaces per-book data
4. **Modular design** — swap embedding model or chunking strategy without touching other components

For very large collections (1000+ books), consider:
- Switching from local Qdrant to Qdrant Cloud (same API, just change the connection)
- Using a dedicated embedding service (e.g., text-embedding-3-small via OpenAI)
- Adding a caching layer (Redis) for frequent queries

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_chatbot.py -v

# With coverage
pip install pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing
```

Tests use **mocked dependencies** — no PostgreSQL or OpenAI key needed.

---

## License

MIT — use freely for study, practice, and spiritual exploration.

---

🤖 Built with [Claude Code](https://claude.ai/code)
