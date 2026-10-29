"""
Ask router: POST /ask — the RAG question-answering endpoint.
"""

import logging
import json
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import QuestionHistory
from ..schemas import AskRequest, AskResponse, SourceReference
from ..config import settings
from ..services.retrieval_service import retrieve, _normalize
from ..services.llm_service import generate_answer, validate_configuration, is_no_answer, LLMError
from ..services.rate_limiter import check_rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ask"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question using the RAG pipeline",
)
def ask_question(
    payload: AskRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    RAG pipeline:
    1. Rate-limit check.
    2. Embed the question and run vector similarity search.
    3. Rerank (dense + IDF-weighted term coverage) and apply relevance gates.
    4. Select evidence sentences; only chunks with evidence become sources.
    5. Generate a grounded answer with the configured LLM (or abstain if
       nothing is relevant). LLM failures are returned as clear errors.
    6. Return structured response with sources.
    """
    # ── 1. Rate limiting ─────────────────────────────────────────────────────
    allowed, retry_after = check_rate_limit(request)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded. Maximum {settings.rate_limit_requests} requests "
                f"per {settings.rate_limit_window} seconds."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    question = payload.question

    # ── 1b. LLM configuration (fail fast with a clear message) ────────────────
    try:
        validate_configuration()
    except LLMError as exc:
        logger.error(f"LLM not configured: {exc.message}")
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    # ── 2-4. Retrieve relevant chunks (embed → vector search → rerank → filter) ──
    try:
        retrieved = retrieve(db, question, top_k=settings.top_k_chunks)
    except Exception as exc:
        logger.error(f"Retrieval failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the question. Please try again.",
        )

    grounding_chunks = []
    source_refs = []
    seen_chunk_ids = set()
    seen_excerpts = set()
    for r in retrieved:
        evidence_text = " ".join(r.evidence)
        excerpt = evidence_text[:300] + ("…" if len(evidence_text) > 300 else "")
        # Safety net: a source is unique by chunk_id, falling back to
        # title + excerpt for the same text stored under different chunk ids.
        excerpt_key = (_normalize(r.title), _normalize(excerpt))
        if r.chunk_id in seen_chunk_ids or excerpt_key in seen_excerpts:
            continue
        seen_chunk_ids.add(r.chunk_id)
        seen_excerpts.add(excerpt_key)
        grounding_chunks.append(
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "title": r.title,
                "content": r.content,
                "evidence": r.evidence,
            }
        )
        source_refs.append(
            SourceReference(
                document_id=r.document_id,
                title=r.title,
                chunk_id=r.chunk_id,
                excerpt=excerpt,
                relevance_score=round(r.score, 4),
            )
        )

    # ── 5. Generate answer ────────────────────────────────────────────────────
    try:
        answer, response_mode = generate_answer(question, grounding_chunks)
    except LLMError as exc:
        logger.error(f"LLM generation failed [{exc.code}]: {exc.message}")
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        raise HTTPException(status_code=exc.status_code, detail=exc.message, headers=headers)

    # If the assistant declined to answer, the retrieved chunks were not
    # actually the answer, so don't present them as sources.
    if is_no_answer(answer):
        source_refs = []

    # ── 6. Persist question history ───────────────────────────────────────────
    try:
        history = QuestionHistory(
            question=question,
            answer=answer,
            source_chunk_ids=json.dumps([r.chunk_id for r in source_refs]),
            response_mode=response_mode,
        )
        db.add(history)
        db.commit()
    except Exception as exc:
        logger.warning(f"Failed to persist question history: {exc}")
        db.rollback()

    logger.info(
        f"Question answered: '{question[:60]}…' "
        f"sources={len(source_refs)} mode={response_mode}"
    )

    return AskResponse(
        answer=answer,
        sources=source_refs,
        question=question,
        response_mode=response_mode,
        retrieval_count=len(source_refs),
    )
