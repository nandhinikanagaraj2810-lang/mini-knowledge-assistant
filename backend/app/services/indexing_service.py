"""
Document indexing: chunk -> embed -> persist chunk rows.

Shared by POST /documents and the startup re-index of stale documents.
"""

import logging
import uuid
from typing import List, Tuple

from sqlalchemy.orm import Session

from ..models import Document, DocumentChunk
from .chunking_service import split_into_chunks
from .embedding_service import embed_texts, embedding_to_bytes, embedding_signature

logger = logging.getLogger(__name__)


def chunk_embedding_text(title: str, content: str) -> str:
    """
    Text that is embedded for a chunk. The document title is included so a
    chunk like "Employees get 10 days per year." still carries its topic
    ("Employee Leave Policy").
    """
    return f"{title}\n{content}"


def build_chunks(doc: Document, text: str) -> Tuple[List[DocumentChunk], list]:
    """Chunk + embed *text* for *doc*. Returns (chunk rows, embeddings matrix)."""
    chunks_text = split_into_chunks(text)
    if not chunks_text:
        return [], []
    embeddings = embed_texts([chunk_embedding_text(doc.title, c) for c in chunks_text])
    signature = embedding_signature()
    rows = [
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=i,
            content=c,
            embedding=embedding_to_bytes(emb),
            embedding_model=signature,
        )
        for i, (c, emb) in enumerate(zip(chunks_text, embeddings))
    ]
    return rows, embeddings


def reindex_stale_documents(db: Session) -> int:
    """
    Re-chunk and re-embed every document whose chunks were produced by a
    different chunker/embedding backend than the one currently active (or by
    an older version of the app that did not record it). Returns the number of
    documents re-indexed.
    """
    signature = embedding_signature()
    count = 0
    for doc in db.query(Document).all():
        stale = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc.id)
            .filter(
                (DocumentChunk.embedding_model.is_(None))
                | (DocumentChunk.embedding_model != signature)
            )
            .count()
        )
        has_chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count()
        if has_chunks and not stale:
            continue
        try:
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
            rows, _ = build_chunks(doc, doc.content)
            db.add_all(rows)
            doc.chunk_count = len(rows)
            doc.status = "indexed" if rows else "error"
            db.commit()
            count += 1
            logger.info(f"Re-indexed document '{doc.title}' ({len(rows)} chunks).")
        except Exception as exc:
            db.rollback()
            logger.error(f"Failed to re-index document {doc.id}: {exc}")
    return count
