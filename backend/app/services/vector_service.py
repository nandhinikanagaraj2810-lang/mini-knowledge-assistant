"""
Vector search service using NumPy cosine similarity.

The index is an in-memory numpy matrix rebuilt from the database on startup.
All chunk embeddings are stored in SQLite as raw bytes; on restart the index
is reconstructed from those stored embeddings so no data is lost.

Falls back to a pure-Python implementation if numpy is not available
(slower, but functional for testing and small datasets).
"""

import logging
import struct
from typing import List, Tuple, Optional
from sqlalchemy.orm import Session

from ..models import DocumentChunk
from ..services.embedding_service import bytes_to_embedding
from ..config import settings

logger = logging.getLogger(__name__)

# Try to import numpy — fall back to pure Python if unavailable
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False
    logger.warning(
        "NumPy not available. Using pure-Python vector search (slower). "
        "Install numpy for production use."
    )


def _dot(a, b):
    """Pure-Python dot product fallback."""
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    """Pure-Python L2 norm fallback."""
    import math
    return math.sqrt(sum(x * x for x in a))


def _cosine_sim_py(query, matrix_rows):
    """Compute cosine similarity between query and each row (pure Python)."""
    q_norm = _norm(query)
    if q_norm == 0:
        return [0.0] * len(matrix_rows)
    q_unit = [x / q_norm for x in query]
    scores = []
    for row in matrix_rows:
        row_norm = _norm(row)
        if row_norm == 0:
            scores.append(0.0)
        else:
            row_unit = [x / row_norm for x in row]
            scores.append(_dot(q_unit, row_unit))
    return scores


class VectorIndex:
    """
    In-memory cosine-similarity index backed by SQLite persistence.
    Uses NumPy when available; falls back to pure Python otherwise.
    """

    def __init__(self):
        self._chunk_ids: List[str] = []
        # NumPy path: np.ndarray; pure-Python path: List[List[float]]
        self._vectors = None

    # ── Build / Rebuild ──────────────────────────────────────────────────────

    def rebuild_from_db(self, db: Session) -> int:
        chunks = db.query(DocumentChunk).filter(DocumentChunk.embedding.isnot(None)).all()
        if not chunks:
            self._chunk_ids = []
            self._vectors = None
            logger.info("Vector index rebuilt: 0 chunks (empty knowledge base).")
            return 0

        ids = []
        vectors = []
        for chunk in chunks:
            try:
                vec = bytes_to_embedding(chunk.embedding)
                ids.append(chunk.id)
                vectors.append(vec)
            except Exception as exc:
                logger.warning(f"Skipping chunk {chunk.id}: {exc}")

        self._chunk_ids = ids
        if _HAS_NUMPY:
            self._vectors = np.stack(vectors)
        else:
            self._vectors = [list(v) for v in vectors]

        logger.info(f"Vector index rebuilt: {len(ids)} chunks loaded from database.")
        return len(ids)

    def add_chunks(self, chunk_ids: List[str], embeddings) -> None:
        if len(chunk_ids) == 0:
            return
        self._chunk_ids.extend(chunk_ids)
        if _HAS_NUMPY:
            if self._vectors is None:
                self._vectors = embeddings.copy()
            else:
                self._vectors = np.vstack([self._vectors, embeddings])
        else:
            rows = [list(e) for e in embeddings]
            if self._vectors is None:
                self._vectors = rows
            else:
                self._vectors.extend(rows)

    def remove_document_chunks(self, chunk_ids_to_remove: set) -> None:
        if self._vectors is None:
            return
        keep_indices = [i for i, cid in enumerate(self._chunk_ids) if cid not in chunk_ids_to_remove]
        if len(keep_indices) == len(self._chunk_ids):
            return
        self._chunk_ids = [self._chunk_ids[i] for i in keep_indices]
        if _HAS_NUMPY:
            self._vectors = self._vectors[keep_indices] if keep_indices else None
        else:
            self._vectors = [self._vectors[i] for i in keep_indices] if keep_indices else None

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding,
        top_k: int = None,
        threshold: float = None,
    ) -> List[Tuple[str, float]]:
        top_k = top_k or settings.top_k_chunks
        threshold = threshold if threshold is not None else settings.relevance_threshold

        if self._vectors is None or len(self._chunk_ids) == 0:
            return []

        if _HAS_NUMPY:
            query = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            scores = self._vectors @ query  # (N,)
            indices = np.where(scores >= threshold)[0]
            if len(indices) == 0:
                return []
            top_indices = indices[np.argsort(scores[indices])[::-1][:top_k]]
            return [(self._chunk_ids[i], float(scores[i])) for i in top_indices]
        else:
            # Pure Python path
            scores = _cosine_sim_py(list(query_embedding), self._vectors)
            filtered = [(i, s) for i, s in enumerate(scores) if s >= threshold]
            filtered.sort(key=lambda x: x[1], reverse=True)
            return [(self._chunk_ids[i], s) for i, s in filtered[:top_k]]

    @property
    def size(self) -> int:
        return len(self._chunk_ids)


# Module-level singleton
_index: Optional[VectorIndex] = None


def get_index() -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex()
    return _index


def rebuild_index(db: Session) -> int:
    """Rebuild the global index from the database. Called at startup."""
    return get_index().rebuild_from_db(db)
