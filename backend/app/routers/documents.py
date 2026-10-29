"""
Documents router: POST /documents and GET /documents.
"""

import logging
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Document, DocumentChunk
from ..schemas import DocumentCreate, DocumentResponse, DocumentListResponse
from ..config import settings
from ..services.indexing_service import build_chunks
from ..services.vector_service import get_index

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a knowledge document",
)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)):
    """
    Ingest a document:
    1. Validate title and content.
    2. Persist document metadata.
    3. Split into chunks.
    4. Generate embeddings.
    5. Persist chunks with embeddings.
    6. Update the in-memory vector index.
    """
    # Size guard
    if len(payload.content) > settings.max_document_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Content exceeds maximum allowed size of "
                f"{settings.max_document_size} characters."
            ),
        )

        # Prevent duplicate documents with the same title and content
    existing_doc = (
        db.query(Document)
        .filter(
            Document.title == payload.title,
            Document.content == payload.content,
        )
        .first()
    )

    if existing_doc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A document with the same title and content already exists.",
        )

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        title=payload.title,
        content=payload.content,
        char_count=len(payload.content),
        status="pending",
        source_type="text",
    )
    db.add(doc)
    db.flush()  # get the ID before committing

    # Chunking + embeddings (title-aware)
    try:
        db_chunks, embeddings = build_chunks(doc, payload.content)
    except Exception as exc:
        logger.error(f"Embedding generation failed: {exc}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding generation failed. Please try again.",
        )
    if not db_chunks:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document content produced no usable chunks.",
        )

    # Persist chunks
    chunk_ids = [c.id for c in db_chunks]
    db.add_all(db_chunks)

    # Update document status
    doc.chunk_count = len(db_chunks)
    doc.status = "indexed"
    db.commit()
    db.refresh(doc)

    # Update in-memory vector index
    get_index().add_chunks(chunk_ids, embeddings)

    logger.info(
        f"Document ingested: id={doc_id} title='{payload.title}' "
        f"chars={doc.char_count} chunks={doc.chunk_count}"
    )
    return doc


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all knowledge documents",
)
def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    total = db.query(Document).count()
    docs = (
        db.query(Document)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return DocumentListResponse(
        documents=docs,
        total=total,
        limit=limit,
        offset=offset,
    )
