"""
SQLAlchemy ORM models for the knowledge assistant.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from .database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(512), nullable=False, index=True)
    content = Column(Text, nullable=False)
    source_type = Column(String(32), default="text")  # "text", "pdf", "txt"
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # Derived metadata
    char_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(String(32), default="indexed")  # "pending", "indexed", "error"

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    # Embeddings stored as raw bytes (numpy float32 array serialised with numpy.tobytes)
    embedding = Column(LargeBinary, nullable=True)
    # Signature of the chunker + embedding backend that produced this row
    embedding_model = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    document = relationship("Document", back_populates="chunks")


class QuestionHistory(Base):
    __tablename__ = "question_history"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    source_chunk_ids = Column(Text, nullable=True)  # JSON list of chunk IDs
    created_at = Column(DateTime, default=_utcnow)
    response_mode = Column(String(32), default="demo")  # "demo" | "llm"
