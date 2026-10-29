"""
Retrieval + relevance filtering for the RAG pipeline.

Pipeline
--------
1. Dense search: cosine similarity between the question embedding and every
   chunk embedding (a wide candidate pool, low threshold).
2. Hybrid rerank: each candidate gets
        score = w_dense * dense_similarity + w_cov * term_coverage
   where term_coverage is the IDF-weighted share of the question's key terms
   that the chunk (title + body) contains. Terms that appear in every chunk
   ("days", "year", ...) therefore count for little, while discriminating terms
   ("sick", "refund") count for a lot.
3. Relevance gates (all must pass):
     - minimum dense similarity (real embedding model) or minimum term
       coverage (lexical fallback embedding),
     - minimum final score,
     - score >= `relative_score_cutoff` * best score  (drops the "other"
       document when one document clearly answers the question).
4. Evidence selection: inside each surviving chunk, split into sentences and
   keep only those that actually answer the question (same hybrid score,
   relative to the best sentence). Chunks with no evidence sentence are
   discarded, so only genuinely relevant sources are returned.

Nothing here is specific to any particular question or document.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from sqlalchemy.orm import Session

from ..config import settings
from ..models import Document, DocumentChunk
from .embedding_service import embed_text, embed_texts, embedding_backend
from .text_utils import split_sentences_with_context, tokenize
from .vector_service import get_index
from .indexing_service import chunk_embedding_text

logger = logging.getLogger(__name__)

NO_ANSWER_MESSAGE = (
    "I couldn't find enough information in the available documents to answer that question."
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    content: str
    score: float
    dense: float
    coverage: float
    evidence: List[str] = field(default_factory=list)


# ── de-duplication helpers ────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Whitespace/case-insensitive form of *text* used for duplicate detection."""
    return " ".join((text or "").split()).casefold()


def dedupe_ranked(items: Sequence, key_fns) -> list:
    """
    Drop repeated items, keeping the first occurrence (so pass them best-first).

    *key_fns* is a list of functions, each mapping an item to a hashable
    identity. An item is a duplicate if ANY of its identities was already seen.
    Distinct chunks stay: they differ in every identity, even when they belong
    to the same document.
    """
    seen = [set() for _ in key_fns]
    kept = []
    for item in items:
        keys = [fn(item) for fn in key_fns]
        if any(k in s for k, s in zip(keys, seen)):
            continue
        for k, s in zip(keys, seen):
            s.add(k)
        kept.append(item)
    return kept


# ── scoring helpers ───────────────────────────────────────────────────────────

def _blend(dense: float, coverage: float) -> float:
    """Combine dense similarity and term coverage into a 0..1 relevance score."""
    if embedding_backend() == "lexical":
        # The lexical embedding is itself keyword-based; weight both equally.
        score = 0.5 * dense + 0.5 * coverage
    else:
        # Real semantic embeddings carry most of the signal.
        score = 0.7 * dense + 0.3 * coverage
    return max(0.0, min(1.0, score))


class _TermStats:
    """IDF statistics over every stored chunk (title + body)."""

    def __init__(self, texts: Sequence[str]):
        self.n = max(len(texts), 1)
        self.df: Dict[str, int] = {}
        for t in texts:
            for tok in set(tokenize(t)):
                self.df[tok] = self.df.get(tok, 0) + 1

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def coverage(self, query_terms: Set[str], text: str) -> float:
        """IDF-weighted fraction of *query_terms* present in *text* (0..1)."""
        if not query_terms:
            return 0.0
        present = set(tokenize(text))
        total = sum(self.idf(t) for t in query_terms)
        if total <= 0:
            return 0.0
        hit = sum(self.idf(t) for t in query_terms if t in present)
        return hit / total


def _dense(a, b) -> float:
    """Cosine similarity of two (already normalised) vectors."""
    try:
        import numpy as np
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)
    except ImportError:  # pragma: no cover
        num = sum(x * y for x, y in zip(a, b))
        den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0
        return num / den


def _passes_gate(dense: float, coverage: float, score: float) -> bool:
    if score < settings.min_final_score:
        return False
    if embedding_backend() == "lexical":
        return coverage >= settings.min_term_coverage
    return dense >= settings.relevance_threshold


# ── public API ────────────────────────────────────────────────────────────────

def retrieve(db: Session, question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
    """Return only the chunks (with evidence sentences) relevant to *question*."""
    top_k = top_k or settings.top_k_chunks
    q_vec = embed_text(question)

    candidate_threshold = 0.05 if embedding_backend() == "lexical" else settings.relevance_threshold * 0.5
    candidates = get_index().search(
        q_vec,
        top_k=settings.candidate_pool_size,
        threshold=candidate_threshold,
    )
    if not candidates:
        return []

    # Corpus statistics (title + body of all chunks) for IDF weighting.
    rows = db.query(DocumentChunk, Document).join(Document, Document.id == DocumentChunk.document_id).all()
    if not rows:
        return []
    by_id = {c.id: (c, d) for c, d in rows}
    stats = _TermStats([chunk_embedding_text(d.title, c.content) for c, d in rows])
    q_terms = set(tokenize(question))

    scored: List[RetrievedChunk] = []
    for chunk_id, dense in candidates:
        if chunk_id not in by_id:
            continue
        chunk, doc = by_id[chunk_id]
        cov = stats.coverage(q_terms, chunk_embedding_text(doc.title, chunk.content))
        score = _blend(dense, cov)
        if not _passes_gate(dense, cov, score):
            continue
        scored.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=doc.id,
                title=doc.title,
                content=chunk.content,
                score=score,
                dense=dense,
                coverage=cov,
            )
        )
    if not scored:
        return []

    scored.sort(key=lambda r: r.score, reverse=True)
    # The same text can be stored under several chunk ids (e.g. a document
    # uploaded more than once). Keep only the best-scoring copy, *before* the
    # top_k cut so copies don't crowd out genuinely different chunks.
    scored = dedupe_ranked(
        scored,
        [lambda r: r.chunk_id, lambda r: (_normalize(r.title), _normalize(r.content))],
    )
    best = scored[0].score
    scored = [r for r in scored if r.score >= settings.relative_score_cutoff * best][:top_k]

    # Evidence sentences. The cutoff is applied across ALL candidate chunks so
    # a weaker sentence from a neighbouring chunk/document cannot ride along
    # with the best answer. Chunks left without evidence are dropped.
    per_chunk = {r.chunk_id: score_sentences(question, r.content, stats, q_terms) for r in scored}
    all_scores = [s[3] for sents in per_chunk.values() for s in sents]
    if not all_scores:
        return []
    cutoff = settings.sentence_relative_cutoff * max(all_scores)

    flat = [
        (score, r.chunk_id, order, text)
        for r in scored
        for (order, text, _heading, score) in per_chunk[r.chunk_id]
        if score >= cutoff
    ]
    flat.sort(key=lambda x: x[0], reverse=True)
    flat = flat[: settings.max_evidence_sentences]

    keep_by_chunk: Dict[str, List] = {}
    for score, chunk_id, order, text in flat:
        keep_by_chunk.setdefault(chunk_id, []).append((order, text, score))

    final: List[RetrievedChunk] = []
    for r in scored:
        picked = keep_by_chunk.get(r.chunk_id)
        if not picked:
            continue
        picked.sort(key=lambda p: p[0])  # document order
        r.evidence = [p[1] for p in picked]
        final.append(r)
    final.sort(key=lambda r: max(s[2] for s in keep_by_chunk[r.chunk_id]), reverse=True)
    # Different chunks (e.g. overlapping ones) may still surface the very same
    # evidence under the same title; show that source only once.
# Remove duplicate evidence even when document titles differ.
    return dedupe_ranked(
    final,
    [
        lambda r: (_normalize(r.title), _normalize(" ".join(r.evidence))),
        lambda r: _normalize(" ".join(r.evidence)),
    ],
)

def score_sentences(
    question: str,
    content: str,
    stats: _TermStats,
    q_terms: Optional[Set[str]] = None,
):
    """
    Score every sentence of *content* against *question*.

    Returns a list of (order, display_text, heading, score) for the sentences
    that pass the relevance gates.
    """
    q_terms = q_terms if q_terms is not None else set(tokenize(question))
    pairs = split_sentences_with_context(content)
    if not pairs:
        return []

    q_vec = embed_text(question)
    # Embed/score the sentence together with its nearest heading for context.
    texts = [f"{h}: {s}" if h else s for s, h in pairs]
    vecs = embed_texts(texts)

    out = []
    for idx, ((sent, heading), vec) in enumerate(zip(pairs, vecs)):
        cov = stats.coverage(q_terms, texts[idx])
        dense = _dense(q_vec, vec)
        score = _blend(dense, cov)
        if _passes_gate(dense, cov, score):
            display = f"{heading}: {sent}" if heading and heading.lower() not in sent.lower() else sent
            out.append((idx, display, heading, score))
    return out
