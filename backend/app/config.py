"""
Application configuration using Pydantic Settings.
All sensitive values are loaded from environment variables or .env file.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path

# backend/.env, independent of the directory the server is started from
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # Application
    app_name: str = "Mini Knowledge Assistant"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database
    database_url: str = "sqlite:///./knowledge_assistant.db"

    # Embedding model (local, no API key required)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_document_size: int = 100_000  # characters

    # Retrieval
    top_k_chunks: int = 5
    # Minimum dense cosine similarity for a chunk to even be a candidate when
    # a real sentence-transformers model is in use.
    relevance_threshold: float = 0.3
    # Minimum final (hybrid) relevance score for a chunk to be used as context.
    min_final_score: float = 0.35
    # Keep only chunks scoring at least this fraction of the best chunk's score.
    relative_score_cutoff: float = 0.6
    # Minimum IDF-weighted share of the question's key terms that a chunk
    # (or answer sentence) must cover to be considered relevant.
    min_term_coverage: float = 0.34
    # Candidates fetched from the vector index before reranking/filtering.
    candidate_pool_size: int = 25
    # Inside a retrieved chunk, keep only sentences scoring at least this
    # fraction of the best sentence (prevents e.g. casual-leave sentences from
    # riding along with a sick-leave answer).
    sentence_relative_cutoff: float = 0.8
    max_evidence_sentences: int = 3

    # LLM provider: "gemini" (default), "openai", "anthropic", or "demo".
    # "demo" is an explicit offline opt-in only; it is never used as a silent fallback.
    llm_provider: str = "gemini"
    # Leave unset to use the provider's default model (see llm_service.DEFAULT_MODELS).
    llm_model: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_timeout: int = 60
    llm_max_retries: int = 2

    # Rate limiting
    rate_limit_requests: int = 20
    rate_limit_window: int = 60  # seconds

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
