"""
Embedding service using sentence-transformers (local, no API key required).
Falls back to a deterministic *lexical* embedding for offline use.

Backends:
  - "sentence-transformers": real semantic embeddings (recommended; install
    `sentence-transformers`).
  - "lexical": feature-hashed bag of stemmed content words + bigrams. Fully
    deterministic and offline. Texts that talk about the same topic share
    vocabulary and therefore score high; unrelated texts score ~0. It is NOT
    truly semantic (it cannot link synonyms) — hybrid reranking in
    retrieval_service compensates by also checking IDF-weighted term coverage.

The previous fallback was a *random* vector per text, which carried no
meaning at all: true matches scored ~0 and the app then fell back to loose
keyword matching, producing mixed/irrelevant answers.
"""


import logging
import zlib
import struct
from typing import List, Union
from ..config import settings
from .text_utils import tokenize

logger = logging.getLogger(__name__)

# Try importing numpy
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False
    logger.warning("NumPy not available. Embeddings will use pure-Python lists.")

_model = None
_use_fallback = False
_load_attempted = False


def _load_model():
    global _model, _use_fallback, _load_attempted
    if _model is not None or _load_attempted:
        return _model
    _load_attempted = True
    if not _HAS_NUMPY:
        logger.warning("Cannot load sentence-transformers without numpy. Using lexical fallback.")
        _use_fallback = True
        return None
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded successfully.")
        _use_fallback = False
    except Exception as exc:
        logger.warning(
            f"Could not load sentence-transformers model ({exc}). "
            "Falling back to the offline lexical embedding (keyword-level, not truly semantic). "
            "Install `sentence-transformers` for real semantic retrieval."
        )
        _model = None
        _use_fallback = True
    return _model


def _hash_embedding_py(text: str, dim: int = 384) -> List[float]:
    """
    Deterministic lexical embedding (pure Python): signed feature hashing of
    stemmed content words and word bigrams, sub-linear term frequency,
    L2-normalised.
    """
    import math
    from collections import Counter

    toks = tokenize(text)
    feats = Counter(toks)
    for a, b in zip(toks, toks[1:]):
        feats[f"{a}_{b}"] += 0.5  # bigrams: weaker than unigrams

    vec = [0.0] * dim
    for feat, count in feats.items():
        h = zlib.crc32(feat.encode("utf-8"))
        idx = h % dim
        sign = 1.0 if (h >> 16) & 1 else -1.0
        vec[idx] += sign * (1.0 + math.log(count)) if count >= 1 else sign * count
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _hash_embedding_np(text: str, dim: int = 384):
    """NumPy wrapper around the lexical embedding."""
    return np.asarray(_hash_embedding_py(text, dim), dtype=np.float32)


def embed_text(text: str):
    """Return a normalised embedding vector for a single text string."""
    _load_model()
    if not _use_fallback and _model is not None:
        embedding = _model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return embedding.astype(np.float32)
    elif _HAS_NUMPY:
        return _hash_embedding_np(text, settings.embedding_dimension)
    else:
        return _hash_embedding_py(text, settings.embedding_dimension)


def embed_texts(texts: List[str]):
    """Return embeddings for a list of texts."""
    _load_model()
    if not _use_fallback and _model is not None:
        embeddings = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
        return embeddings.astype(np.float32)
    elif _HAS_NUMPY:
        return np.stack([_hash_embedding_np(t, settings.embedding_dimension) for t in texts])
    else:
        return [_hash_embedding_py(t, settings.embedding_dimension) for t in texts]


def embedding_to_bytes(embedding) -> bytes:
    """Serialise an embedding (numpy array or list) to bytes for DB storage."""
    if _HAS_NUMPY and hasattr(embedding, 'tobytes'):
        return embedding.astype(np.float32).tobytes()
    else:
        # Encode as list of 32-bit floats
        return struct.pack(f'{len(embedding)}f', *embedding)


def bytes_to_embedding(data: bytes):
    """Deserialise bytes back to an embedding (numpy array or list)."""
    if _HAS_NUMPY:
        return np.frombuffer(data, dtype=np.float32).copy()
    else:
        n = len(data) // 4
        return list(struct.unpack(f'{n}f', data))


def is_using_fallback() -> bool:
    _load_model()
    return _use_fallback


def embedding_backend() -> str:
    """"sentence-transformers" or "lexical"."""
    return "lexical" if is_using_fallback() else "sentence-transformers"


def embedding_signature() -> str:
    """
    Identifies how stored embeddings (and chunks) were produced. When it
    changes, documents are automatically re-chunked and re-embedded at startup.
    """
    from .chunking_service import CHUNKER_VERSION
    if is_using_fallback():
        return f"lexical-v2|{settings.embedding_dimension}|chunk-{CHUNKER_VERSION}"
    return f"st:{settings.embedding_model}|chunk-{CHUNKER_VERSION}"
