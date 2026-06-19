"""
Configuration management for the Spiritual Assistant.

Loads all settings from environment variables (via .env file).
Provides a single, typed settings object for the entire application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────
# The .env file lives one directory above app/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class Settings:
    """Typed application settings loaded from environment variables."""

    # ── Paths ────────────────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    books_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "books")
    embeddings_cache_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "embeddings"
    )

    # ── Qdrant (Local Embedded) ──────────────────────────────────────────
    qdrant_path: str = field(
        default_factory=lambda: os.getenv("QDRANT_PATH", "./qdrant_data")
    )
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "spiritual_books")
    )
    # Resolved absolute path (relative paths are relative to project root)
    @property
    def qdrant_abs_path(self) -> str:
        p = Path(self.qdrant_path)
        if not p.is_absolute():
            p = self.project_root / p
        return str(p.resolve())

    # ── OpenRouter ────────────────────────────────────────────────────────
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_MODEL", "google/gemini-2.5-flash"
        )
    )
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
    )

    # ── Embedding Model ──────────────────────────────────────────────────
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "384"))
    )

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size_words: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE_WORDS", "500"))
    )
    chunk_overlap_words: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_WORDS", "100"))
    )

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k_results: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "5"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
    )

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # ── API Key Authentication ────────────────────────────────────────────
    api_key: str = field(
        default_factory=lambda: os.getenv("API_KEY", "")
    )

    # ── MySQL ─────────────────────────────────────────────────────────────
    mysql_host: str = field(
        default_factory=lambda: os.getenv("MYSQL_HOST", "localhost")
    )
    mysql_port: int = field(
        default_factory=lambda: int(os.getenv("MYSQL_PORT", "3306"))
    )
    mysql_dbname: str = field(
        default_factory=lambda: os.getenv("MYSQL_DBNAME", "spiritual_chatbot")
    )
    mysql_user: str = field(
        default_factory=lambda: os.getenv("MYSQL_USER", "root")
    )
    mysql_password: str = field(
        default_factory=lambda: os.getenv("MYSQL_PASSWORD", "")
    )

    def validate(self) -> None:
        """Raise ValueError if required settings are missing."""
        if not self.openrouter_api_key or "your-" in self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Please configure it in your .env file."
            )


# ── Singleton settings instance ──────────────────────────────────────────
settings = Settings()
