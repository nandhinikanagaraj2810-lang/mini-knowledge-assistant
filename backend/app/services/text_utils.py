"""
Shared text helpers: tokenisation, light stemming, sentence splitting.

These are deliberately dependency-free so they work in every environment
(with or without sentence-transformers).
"""

import re
from typing import List, Tuple

# Generic English function words + question words that carry no topical meaning.
STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from into onto over under
    as is are was were be been being am do does did done doing have has had having
    it its this that these those there here their theirs our ours your yours my mine
    he she they them his her him we us you i me who whom whose which what when where
    why how can could should would will shall may might must not no nor so than too
    very just also about per each any all some such only own same other more most
    many much few several get gets got give given eligible entitled
    long take takes need needs tell please know whether there's let
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def stem(token: str) -> str:
    """Very light suffix stripper (refunds/refunded/refunding -> refund)."""
    t = token.replace("’", "'")
    if t.endswith("'s"):
        t = t[:-2]
    if len(t) <= 3 or t.isdigit():
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("sses"):
        return t[:-2]
    if t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("es") and len(t) > 4 and t[-3] in "sxz":
        t = t[:-2]
    elif t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        t = t[:-1]
    if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "ls":
        t = t[:-1]
    return t


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    """Lower-case, strip stopwords and stem."""
    out = []
    for raw in _WORD_RE.findall(text.lower()):
        if not keep_stopwords and raw in STOPWORDS:
            continue
        out.append(stem(raw))
    return out


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith(("-", "*", "•")):
        return False
    if s.endswith(":") and len(s.split()) <= 10:
        return True
    words = s.split()
    if len(words) > 6:
        return False
    if s.endswith((".", "!", "?")) or ":" in s or any(ch.isdigit() for ch in s):
        return False
    return True


def split_sections(text: str) -> List[str]:
    """
    Split text into paragraph-like blocks (separated by blank lines).
    Short heading-only blocks are merged into the block that follows them so a
    heading ("Sick Leave") always travels with its body.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if b.strip()]
    merged: List[str] = []
    pending_heading = ""
    for b in blocks:
        if "\n" not in b and _is_heading(b):
            pending_heading = (pending_heading + "\n" + b).strip()
            continue
        if pending_heading:
            b = pending_heading + "\n" + b
            pending_heading = ""
        merged.append(b)
    if pending_heading:
        if merged:
            merged[-1] = merged[-1] + "\n" + pending_heading
        else:
            merged.append(pending_heading)
    return merged


def split_sentences_with_context(text: str) -> List[Tuple[str, str]]:
    """
    Split text into (sentence, context) pairs.

    `context` is the nearest preceding heading line (if any), so scoring can use
    "Sick Leave" as context for the sentence "Employees get 10 days per year."
    Each line is treated as its own unit, and lines are further split on
    sentence-ending punctuation.
    """
    pairs: List[Tuple[str, str]] = []
    heading = ""
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        if _is_heading(line) and not line.startswith(("-", "*", "•")):
            heading = line.rstrip(":").strip()
            continue
        # "Sick leave: employees get 10 days." -> keep the label as context too
        for sent in _SENT_SPLIT_RE.split(line):
            sent = sent.strip()
            if sent:
                pairs.append((sent, heading))
    return pairs
