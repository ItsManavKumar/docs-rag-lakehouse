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
    # seen in DuSpec+ (Dulux/Cabot's) and Selleys data sheets
    "introduction", "description and image", "product information", "technical details",
    "technical features", "how to use", "approvals & standards", "approvals and standards",
    "precautions and limitations", "typical properties", "application data",
    "application details", "recommended systems", "substrates", "additional information",
    "technical specifications", "environmental", "sustainability", "approvals",
    "standards & certificates", "standards and certificates",
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
    ("accidental", "Spills"), ("spill", "Spills"),
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
    ("transport", "Transport"), ("dangerous good", "Transport"), ("regulatory", "Regulatory"),
    ("identification", "Identification"),
    ("cas no", "Composition"), ("proportion", "Composition"),
    ("appl", "Application"), ("direction", "Application"), ("how to", "Application"),
    ("thinning", "Application"),
    ("use", "Uses"), ("suitable", "Uses"), ("where", "Uses"),
    ("introduction", "Product description"),
    ("description", "Product description"), ("feature", "Product description"),
    ("benefit", "Product description"),
    ("colour", "Colours and finish"), ("finish", "Colours and finish"),
    ("sheen", "Colours and finish"), ("tint", "Colours and finish"),
    ("safety", "Safety"), ("precaution", "Safety"), ("limitation", "Limitations"),
    ("pack", "Packaging"), ("maintenance", "Maintenance"),
    ("approv", "Approvals"), ("standard", "Approvals"), ("certif", "Approvals"),
    ("product information", "Technical data"),
)

_SDS_NUMBERED = re.compile(
    r"^(?:section\s*)?(\d{1,2})\s*[.:)\-]?\s+([A-Za-z][A-Za-z0-9 ,&/()'\-]{3,90}?)\s*:?$", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


_CANON_RX = [(re.compile(r"\b" + re.escape(k)), label) for k, label in CANONICAL]
_INLINE_HEADING = re.compile(r"^([A-Z][A-Za-z&/()' \-]{2,45}?)\s*:\s+(.+)$")
_BULLET = re.compile(r"^[\u2022\u25cf\u25aa\-\*\d]+[.)]?\s")
_SMALL_WORDS = {"and", "or", "of", "for", "to", "the", "a", "an", "in", "on", "with", "&"}


def canonical_section(heading: str) -> str:
    h = heading.lower()
    for rx, label in _CANON_RX:
        if rx.search(h):
            return label
    return "General"


def _is_title_case(s: str) -> bool:
    words = [w for w in re.split(r"\s+", s) if w]
    return bool(words) and all(w[0].isupper() or w.lower() in _SMALL_WORDS or not w[0].isalpha()
                               for w in words)


def split_inline_heading(line: str) -> tuple[str, str] | None:
    """'Uses: Use Aquanamel on doors and trim ...' -> ('Uses', 'Use Aquanamel on ...').

    Only when the prefix is a short, title-case phrase with a section keyword AND the rest
    is a real sentence (>= 8 words). Short 'Recoat: 2 hours' lines are table rows, not headings.
    """
    m = _INLINE_HEADING.match(line.strip())
    if not m:
        return None
    head, rest = m.group(1).strip(), m.group(2).strip()
    if len(head.split()) > 5 or len(rest.split()) < 8 or not _is_title_case(head):
        return None
    if head.lower() in TDS_HEADINGS or canonical_section(head) != "General":
        return head, rest
    return None


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
    if (len(words) <= 5 and not _BULLET.match(s) and _is_title_case(bare)
            and not re.search(r"[\d.;,!?]", bare) and canonical_section(bare) != "General"):
        return bare
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
                continue
            inline = split_inline_heading(line)
            if inline:
                sections.append({"heading": inline[0], "lines": [(p["page"], inline[1])]})
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
