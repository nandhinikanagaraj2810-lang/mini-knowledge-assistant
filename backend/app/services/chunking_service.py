"""
Text chunking service.

Splits document content into chunks for embedding and retrieval.

Chunks follow the document's own structure instead of cutting at arbitrary
character offsets:

  1. The text is split into sections (blank-line separated blocks, with short
     headings glued to the block that follows them).
  2. Each section that fits in `chunk_size` becomes ONE chunk, so unrelated
     topics (e.g. "Sick Leave" vs "Casual Leave" sections) are never fused
     into the same chunk just because they were close together.
  3. Oversized sections are packed sentence-by-sentence, with a small
     sentence-level overlap between consecutive chunks.
"""

from typing import List
from ..config import settings
from .text_utils import split_sections, _SENT_SPLIT_RE

# Bump when the chunking algorithm changes so stored documents get re-indexed.
CHUNKER_VERSION = "v2"


def _hard_split(text: str, size: int) -> List[str]:
    return [text[i:i + size].strip() for i in range(0, len(text), size) if text[i:i + size].strip()]


def _pack_units(units: List[str], chunk_size: int, overlap_chars: int) -> List[str]:
    """Greedily pack sentence/line units into chunks of at most chunk_size chars."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for unit in units:
        if len(unit) > chunk_size:
            # flush, then hard-split the monster unit
            if current:
                chunks.append(" ".join(current).strip())
                current, current_len = [], 0
            chunks.extend(_hard_split(unit, chunk_size))
            continue

        added = len(unit) + (1 if current else 0)
        if current and current_len + added > chunk_size:
            chunks.append(" ".join(current).strip())
            # carry trailing sentences as overlap
            carry: List[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) + 1 > overlap_chars:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            current = carry
            current_len = sum(len(c) + 1 for c in current)
            added = len(unit) + (1 if current else 0)
        current.append(unit)
        current_len += added

    if current:
        tail = " ".join(current).strip()
        if tail and (not chunks or tail != chunks[-1]):
            chunks.append(tail)
    return chunks


def split_into_chunks(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[str]:
    """
    Split *text* into structure-aware chunks.

    Parameters
    ----------
    text        : The source text to split.
    chunk_size  : Maximum number of characters per chunk (default: settings.chunk_size).
    chunk_overlap: Characters of sentence-level overlap used only when an
                   oversized section has to be split (default: settings.chunk_overlap).

    Returns
    -------
    A list of non-empty text strings.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    chunks: List[str] = []
    for section in split_sections(text):
        if len(section) <= chunk_size:
            chunks.append(section)
            continue
        units: List[str] = []
        for line in section.split("\n"):
            line = line.strip()
            if not line:
                continue
            units.extend(s.strip() for s in _SENT_SPLIT_RE.split(line) if s.strip())
        chunks.extend(_pack_units(units, chunk_size, chunk_overlap))

    return [c for c in chunks if c.strip()]
