"""
Mini Knowledge Assistant — FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .database import init_db, SessionLocal
from .services.vector_service import rebuild_index
from .services.indexing_service import reindex_stale_documents
from .routers import documents, ask
from .schemas import HealthResponse
from .models import Document

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise DB and rebuild vector index from persisted data."""
    logger.info("Starting Mini Knowledge Assistant …")
    init_db()
    db = SessionLocal()
    try:
        n_reindexed = reindex_stale_documents(db)
        if n_reindexed:
            logger.info(f"Re-indexed {n_reindexed} document(s) with the current chunker/embedder.")
        n = rebuild_index(db)
        logger.info(f"Vector index ready with {n} chunk(s).")
    finally:
        db.close()
    yield
    logger.info("Shutting down.")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "A RAG-powered internal knowledge assistant. "
        "Upload documents and ask questions grounded in your knowledge base."
    ),
    lifespan=lifespan,
)

# CORS — allow the Vite dev server and same-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(documents.router)
app.include_router(ask.router)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(request: Request):
    db = SessionLocal()
    try:
        doc_count = db.query(Document).count()
    finally:
        db.close()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        document_count=doc_count,
    )


# ── Global exception handler (no stack traces to clients) ────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."},
    )
