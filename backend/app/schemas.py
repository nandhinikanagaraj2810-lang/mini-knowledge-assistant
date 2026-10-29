"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime


# ─── Document Schemas ────────────────────────────────────────────────────────

class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=512, description="Document title")
    content: str = Field(..., min_length=1, description="Document text content")

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Title must not be blank.")
        return v.strip()

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content must not be blank.")
        return v


class DocumentResponse(BaseModel):
    id: str
    title: str
    status: str
    char_count: int
    chunk_count: int
    created_at: datetime
    source_type: str

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    limit: int
    offset: int


# ─── Ask / RAG Schemas ───────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Natural language question")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be blank.")
        return v.strip()


class SourceReference(BaseModel):
    document_id: str
    title: str
    chunk_id: str
    excerpt: str
    relevance_score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    question: str
    response_mode: str  # "gemini" | "openai" | "anthropic" | "demo" | "none" (abstained, no LLM call)
    retrieval_count: int


# ─── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    document_count: int
