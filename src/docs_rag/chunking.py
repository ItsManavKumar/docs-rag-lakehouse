"""Section-aware chunking (Silver layer).

Fixed-size chunking cuts "Recoat time: 2 hours" away from the heading and the
product it belongs to. Data sheets are highly structured, so we:
  1. detect section headings (SDS numbered sections, known TDS headings,
     short ALL-CAPS / "Heading:" lines),
  2. keep each section together, splitting only if it exceeds max_words
     (with overlap),
  3. prefix every chunk's *embedding text* with brand | product | doc type |
     section, so two near-identical products produce distinguishable vectors.
"""
from __future__ import annotations

import hashlib
import re

SDS_SECTION_WORDS = (
    "identification", "hazard", "composition", "ingredient", "first aid", "fire",
    "accidental", "handling", "storage", "exposure", "personal protection",
    "physical", "chemical properties", "stability", "reactivity", "toxicolog",
    "ecolog", "disposal", "transport", "regulatory", "other information",
)

TDS_HEADINGS = (
    "product description", "description", "features", "features and benefits",
    "benefits", "recommended uses", "uses", "where to use", "suitable surfaces",
    "surface preparation", "preparation", "priming", "primer", "application",
    "application method", "application equipment", "application conditions",
    "how to apply", "directions for use", "directions", "drying time", "drying times",
    "dry time", "recoat time", "recoat", "coverage", "spreading rate",
    "spread rate", "clean up", "cleanup", "cleaning", "storage", "shelf life",
    "technical data", "technical information", "specifications", "product data",
    "properties", "physical properties", "colours", "colour", "finish", "sheen",
    "limitations", "precautions", "safety", "safety precautions", "health and safety",
    "packaging", "pack sizes", "thinning", "tinting", "maintenance", "important notes",
    "warranty", "disclaimer", "systems", "paint system", "curing", "cure time",
)

# keyword -> canonical label, first match wins (order matters)
CANONICAL = (
    ("recoat", "Recoat time"),
    ("dry", "Drying time"), ("cur", "Drying time"),
    ("surface prep", "Surface preparation"), ("preparation", "Surface preparation"),
    ("prim", "Surface preparation"),
    ("coverage", "Coverage"), ("spread", "Coverage"),
    ("clean", "Clean up"),
    ("first aid", "First aid"),
    ("fire", "Fire fighting"),
    ("accidental", "Spills"),
    ("handling", "Handling and storage"), ("storage", "Handling and storage"),
    ("shelf", "Handling and storage"),
    ("exposure", "Exposure controls / PPE"), ("protection", "Exposure controls / PPE"),
    ("hazard", "Hazards"),
    ("composition", "Composition"), ("ingredient", "Composition"),
    ("physical", "Physical properties"), ("technical", "Technical data"),
    ("specification", "Technical data"), ("properties", "Technical data"),
    ("product data", "Technical data"),
    ("stability", "Stability"), ("reactivity", "Stability"),
    ("toxicolog", "Toxicology"), ("ecolog", "Ecology"), ("disposal", "Disposal"),
    ("transport", "Transport"), ("regulatory", "Regulatory"),
    ("identification", "Identification"),
    ("appl", "Application"), ("direction", "Application"), ("how to", "Application"),
    ("thinning", "Application"),
    ("use", "Uses"), ("suitable", "Uses"), ("where", "Uses"),
    ("description", "Product description"), ("feature", "Product description"),
    ("benefit", "Product description"),
    ("colour", "Colours and finish"), ("finish", "Colours and finish"),
    ("sheen", "Colours and finish"), ("tint", "Colours and finish"),
    ("safety", "Safety"), ("precaution", "Safety"), ("limitation", "Limitations"),
    ("pack", "Packaging"),
)

_SDS_NUMBERED = re.compile(
    r"^(?:section\s*)?(\d{1,2})\s*[.:)\-]?\s+([A-Za-z][A-Za-z0-9 ,&/()'\-]{3,90}?)\s*:?$", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def canonical_section(heading: str) -> str:
    h = heading.lower()
    for key, label in CANONICAL:
        if key in h:
            return label
    return "General"


def detect_heading(line: str) -> str | None:
    """Return a cleaned heading if `line` looks like a section heading, else None."""
    s = line.strip()
    if not s or len(s) > 95:
        return None
    m = _SDS_NUMBERED.match(s)
    if m and 1 <= int(m.group(1)) <= 16:
        title = m.group(2).lower()
        if any(w in title for w in SDS_SECTION_WORDS):
            return f"Section {int(m.group(1))}: {m.group(2).strip().title()}"
    bare = s.rstrip(":").strip()
    words = bare.split()
    if not words or len(words) > 7:
        return None
    if ":" in bare:  # "Recoat: 2 hours" is content (a table row), not a heading
        return None
    if bare.lower() in TDS_HEADINGS:
        return bare.title() if bare.isupper() else bare
    letters = [c for c in bare if c.isalpha()]
    digit_share = sum(c.isdigit() for c in bare) / max(len(bare), 1)
    if len(letters) >= 5 and digit_share < 0.1 and not bare.endswith("."):
        if bare.isupper() and len(words) <= 6:
            return bare.title()
        if s.endswith(":") and bare.istitle():
            return bare
    return None


def _split_long_line(line: str, max_words: int) -> list[str]:
    if len(line.split()) <= max_words:
        return [line]
    out, cur = [], []
    for sent in _SENTENCE_SPLIT.split(line):
        w = sent.split()
        if cur and len(cur) + len(w) > max_words:
            out.append(" ".join(cur))
            cur = []
        while len(w) > max_words:  # a single enormous "sentence"
            out.append(" ".join(w[:max_words]))
            w = w[max_words:]
        cur.extend(w)
    if cur:
        out.append(" ".join(cur))
    return out


def sectionize(pages: list[dict]) -> list[dict]:
    """[{page, text}] -> [{heading, lines:[(page, text)]}] across page breaks."""
    sections = [{"heading": "Overview", "lines": []}]
    for p in sorted(pages, key=lambda x: x["page"]):
        for raw in p["text"].split("\n"):
            line = raw.strip()
            if not line:
                continue
            h = detect_heading(line)
            if h:
                sections.append({"heading": h, "lines": []})
            else:
                sections[-1]["lines"].append((p["page"], line))
    return [s for s in sections if s["lines"]]


def _pack(lines: list[tuple[int, str]], max_words: int, overlap: int) -> list[dict]:
    units = [(pg, piece) for pg, ln in lines for piece in _split_long_line(ln, max_words)]
    chunks, cur, cur_pages = [], [], []
    for pg, text in units:
        n = len(text.split())
        if cur and sum(len(t.split()) for t in cur) + n > max_words:
            chunks.append({"lines": cur, "pages": cur_pages})
            tail = " ".join(" ".join(cur).split()[-overlap:]) if overlap else ""
            cur, cur_pages = ([tail] if tail else []), ([cur_pages[-1]] if tail else [])
        cur.append(text)
        cur_pages.append(pg)
    if cur:
        chunks.append({"lines": cur, "pages": cur_pages})
    return chunks


def chunk_document(pages: list[dict], meta: dict, max_words: int = 220,
                   overlap_words: int = 40, min_words: int = 12) -> list[dict]:
    """Chunk one document. `meta` needs file, brand, product, doc_type."""
    raw: list[dict] = []
    for sec in sectionize(pages):
        for piece in _pack(sec["lines"], max_words, overlap_words):
            raw.append({"section": sec["heading"], "text": "\n".join(piece["lines"]),
                        "page_start": min(piece["pages"]), "page_end": max(piece["pages"])})
    # merge fragments that are too small to be useful on their own
    merged: list[dict] = []
    for c in raw:
        n = len(c["text"].split())
        if merged and n < min_words and len(merged[-1]["text"].split()) + n <= max_words + min_words:
            prev = merged[-1]
            if prev["section"] != c["section"]:
                prev["text"] += f"\n{c['section']}: {c['text']}"
            else:
                prev["text"] += "\n" + c["text"]
            prev["page_end"] = max(prev["page_end"], c["page_end"])
        else:
            merged.append(dict(c))
    # a tiny leading fragment (e.g. just the title) has nothing before it to merge into
    if len(merged) > 1 and len(merged[0]["text"].split()) < min_words:
        first, nxt = merged.pop(0), merged[0]
        nxt["text"] = first["text"] + "\n" + nxt["text"]
        nxt["page_start"] = min(first["page_start"], nxt["page_start"])
    out = []
    for i, c in enumerate(merged):
        header = f"{meta['brand']} | {meta['product']} | {meta['doc_type']} | {c['section']}"
        cid = hashlib.sha1(f"{meta['file']}::{i}::{c['text'][:200]}".encode()).hexdigest()[:16]
        out.append({
            "chunk_id": cid,
            "chunk_index": i,
            "file": meta["file"],
            "brand": meta["brand"],
            "product": meta["product"],
            "doc_type": meta["doc_type"],
            "section": c["section"],
            "section_canonical": canonical_section(c["section"]),
            "page_start": int(c["page_start"]),
            "page_end": int(c["page_end"]),
            "text": c["text"],
            "embed_text": f"{header}\n{c['text']}",
            "n_words": len(c["text"].split()),
        })
    return out
