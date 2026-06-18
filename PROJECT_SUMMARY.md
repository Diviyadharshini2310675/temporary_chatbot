# Spiritual Assistant — Project Summary

## Overview

A RAG (Retrieval-Augmented Generation) powered chatbot that answers spiritual questions strictly from a set of uploaded PDF books and a pre-built Q&A database. The system never halluccinates — it only responds based on content from the ingested books or stored Q&A pairs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI (Python) |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 (384-dim, local) |
| Vector Database | Qdrant (local embedded mode, no server needed) |
| LLM / Answer Generation | OpenRouter API (Gemini 2.5 Flash) |
| PDF Processing | pypdf |
| Q&A Database | PostgreSQL (psycopg2) |
| DOCX Import | python-docx |
| Frontend | HTML + CSS + Vanilla JavaScript |
| Configuration | python-dotenv |
| Data Validation | Pydantic v2 |
| Testing | pytest, pytest-asyncio, httpx |
| Server | Uvicorn (ASGI) |

---

## Project Structure

```
spiritual-chatbot/
│
├── app/                        # Core application modules
│   ├── main.py                 # FastAPI app, all API endpoints, lifespan
│   ├── config.py               # Settings dataclass, loads from .env
│   ├── models.py               # Pydantic request/response models
│   ├── embeddings.py           # Sentence transformer model, encode query/chunks
│   ├── ingest.py               # PDF ingestion pipeline (extract → chunk → embed → store)
│   ├── retrieve.py             # Semantic retrieval with deduplication
│   ├── vectordb.py             # Qdrant client, CRUD, similarity search
│   ├── chatbot.py              # RAG prompt building, OpenRouter LLM call
│   ├── qa_service.py           # PostgreSQL Q&A lookup and user question saving
│   ├── db.py                   # PostgreSQL connection pool (psycopg2)
│   └── __init__.py
│
├── static/
│   └── index.html              # Full frontend UI (chat interface)
│
├── books/                      # Place PDF books here for ingestion
│   └── qa_data/
│       └── qa_file.docx        # DOCX file with Q&A pairs for import
│
├── embeddings/                 # Cached sentence-transformer model files
├── qdrant_data/                # Local Qdrant vector database storage
│
├── tests/
│   ├── conftest.py             # Shared fixtures, mocks for all tests
│   ├── test_api.py             # FastAPI endpoint tests
│   ├── test_chatbot.py         # Chatbot prompt and answer generation tests
│   └── test_ingest.py          # PDF ingestion and chunking tests
│
├── import_qa_docx.py           # CLI script to import Q&A pairs from DOCX to PostgreSQL
├── requirements.txt            # All Python dependencies
├── .env                        # Environment variables (API keys, DB config)
└── README.md                   # Original readme
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the frontend chat UI |
| GET | `/health` | Health check — Qdrant status + chunk count |
| POST | `/ingest` | Ingest all PDFs from books/ into Qdrant |
| POST | `/chat` | Ask a question, get RAG-powered answer |
| POST | `/search` | Retrieval only — returns top chunks, no LLM call |

---

## Overall Flow

### Ingestion Flow (run once per book update)

```
PDF files in books/
       ↓
Extract text page by page (pypdf)
       ↓
Clean text (remove artefacts, normalise whitespace)
       ↓
Split into 500-word chunks with 100-word overlap
       ↓
Encode chunks → 384-dim vectors (all-MiniLM-L6-v2)
       ↓
Store vectors + metadata in Qdrant (local embedded)
```

### Chat Flow (every user question)

```
User Question
       ↓
Check PostgreSQL qa_dataset (exact MD5 hash match)
       ↓ (if found)
Return stored answer instantly ← (no LLM call)
       ↓ (if not found)
Encode question → 384-dim vector
       ↓
Cosine similarity search in Qdrant (top 5 chunks)
       ↓
Deduplicate chunks by chunk_text
       ↓
Check similarity threshold (≥ 0.35)
       ↓ (below threshold)
Return "not found" message
       ↓ (above threshold)
Build RAG prompt with retrieved passages
       ↓
Send to OpenRouter (Gemini 2.5 Flash)
       ↓
Return answer + source citations to user
```

### Q&A Import Flow

```
DOCX file (3500+ Q&A pairs)
       ↓
python import_qa_docx.py --file qa_data/qa_file.docx
       ↓
Parse Q&A pairs (labeled Q:/A: or alternating paragraphs)
       ↓
Normalize question (strip + lowercase)
       ↓
Generate MD5 hash of normalized question
       ↓
UPSERT into PostgreSQL qa_dataset (ON CONFLICT DO NOTHING)
       ↓
Print import statistics
```

---

## Environment Variables (.env)

| Variable | Description | Default |
|---|---|---|
| QDRANT_PATH | Path to local Qdrant storage | ./qdrant_data |
| QDRANT_COLLECTION | Collection name | spiritual_books |
| OPENROUTER_API_KEY | OpenRouter API key | — |
| OPENROUTER_MODEL | LLM model to use | google/gemini-2.5-flash |
| EMBEDDING_MODEL | Sentence transformer model | sentence-transformers/all-MiniLM-L6-v2 |
| EMBEDDING_DIMENSION | Vector dimensions | 384 |
| CHUNK_SIZE_WORDS | Words per chunk | 500 |
| CHUNK_OVERLAP_WORDS | Overlap between chunks | 100 |
| TOP_K_RESULTS | Chunks to retrieve per query | 5 |
| SIMILARITY_THRESHOLD | Minimum cosine similarity | 0.35 |
| PG_HOST | PostgreSQL host | localhost |
| PG_PORT | PostgreSQL port | 5433 |
| PG_DBNAME | PostgreSQL database name | spiritual_chatbot |
| PG_USER | PostgreSQL user | postgres |
| PG_PASSWORD | PostgreSQL password | — |

---

## PostgreSQL Tables

| Table | Purpose |
|---|---|
| qa_dataset | Pre-stored Q&A pairs from DOCX import |
| user_questions | Every question asked by users (deduped) |
| chat_history | Full chat history log |

---

## How to Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Configure .env** — fill in OPENROUTER_API_KEY and PG_PASSWORD

**3. Start the server:**
```bash
set PYTHONHOME=
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**4. Ingest books** — place PDFs in books/ then call:
```
POST http://localhost:8000/ingest
```

**5. Import Q&A pairs from DOCX:**
```bash
python import_qa_docx.py --file "books/qa_data/qa_file.docx" --source-book "AiR Spiritual Books"
```

**6. Open the UI:**
```
http://localhost:8000
```

---

## Key Design Decisions

- **Qdrant local embedded mode** — no external server needed, data persists on disk
- **PostgreSQL Q&A cache** — instant answers for known questions, saves LLM API cost
- **MD5 hash deduplication** — avoids PostgreSQL btree index limits on long text
- **Chunk deduplication in retrieval** — prevents same page appearing multiple times in sources
- **Similarity threshold (0.35)** — filters irrelevant results before LLM call
- **Single uvicorn worker** — required for Qdrant local embedded mode (no concurrent access)
