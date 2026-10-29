"""
Database setup using SQLAlchemy with SQLite.
Provides sync and async session factories.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import settings
import logging

logger = logging.getLogger(__name__)

# SQLite engine - connect_args needed for SQLite threading
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they do not exist (and apply tiny additive migrations)."""
    from . import models  # noqa: F401 – import needed to register models
    from sqlalchemy import inspect, text
    Base.metadata.create_all(bind=engine)
    try:
        cols = {c["name"] for c in inspect(engine).get_columns("document_chunks")}
        if "embedding_model" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE document_chunks ADD COLUMN embedding_model VARCHAR(128)"))
            logger.info("Migrated document_chunks: added embedding_model column.")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Column migration check failed: {exc}")
    logger.info("Database tables initialised.")
