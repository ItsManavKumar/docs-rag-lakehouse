"""Text cleaning and de-duplication helpers (Silver layer)."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
_PAGE_NO = re.compile(r"^\s*(page\s*)?\d+\s*(of|/)\s*\d+\s*$", re.I)


# icon fonts (DuSpec+ data sheets) put glyphs in the Unicode private use area:
# "Clean Up \ue905 Water" -> "Clean Up Water"
_PRIVATE_USE = re.compile(r"[\ue000-\uf8ff]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _PRIVATE_USE.sub(" ", text)
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = text.replace("­", "")                       # soft hyphen
    # re-join words hyphenated across a line break ("pre-\nparation"); letters only,
    # so ranges like "2-\n4 hours" are left alone
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_TRAILING_PAGE_NO = re.compile(r"\s*\bpage\s*\d+\s*(of|/)\s*\d+\s*$", re.I)


def _line_key(line: str) -> str:
    """Compare lines with digits masked, so 'Page 1 of 3' and 'Page 2 of 3' match."""
    return re.sub(r"\d+", "#", line.strip().lower())


def repeated_lines(pages: list[str], min_share: float = 0.6) -> set[str]:
    """Keys of lines that appear on >= min_share of a document's pages (headers/footers).
    Needs at least 2 pages; a line must repeat on at least 2 of them."""
    if len(pages) < 2:
        return set()
    counts = Counter()
    for p in pages:
        counts.update({_line_key(ln) for ln in p.split("\n") if ln.strip()})
    need = max(2, min_share * len(pages))
    return {k for k, c in counts.items() if c >= need and len(k) < 140}


def strip_noise_lines(text: str, drop: set[str]) -> str:
    keep = []
    for ln in text.split("\n"):
        s = ln.strip()
        if _line_key(s) in drop or _PAGE_NO.match(s):
            continue
        s = _TRAILING_PAGE_NO.sub("", s)   # "Acme Pty Ltd ... Page 1 of 2" -> "Acme Pty Ltd ..."
        if s:
            keep.append(s)
    return "\n".join(keep).strip()


def content_hash(text: str) -> str:
    """Hash of text with case, whitespace and punctuation removed - catches
    the same PDF uploaded twice under different names, or boilerplate
    paragraphs repeated verbatim across documents."""
    key = re.sub(r"[\W_]+", "", (text or "").lower())
    return hashlib.sha256(key.encode()).hexdigest()[:16]
