"""Detect which product / brand a question is about, for metadata-filtered retrieval.

Problem this solves: data sheets for near-identical products ("Wash&Wear Low
Sheen" vs "Wash&Wear Kitchen & Bathroom") produce near-identical embeddings,
so pure vector search happily returns the wrong product's recoat time. If the
question names a product, we restrict the search to that product's chunks.
"""
from __future__ import annotations

import re

# words that don't identify a product on their own
GENERIC = {
    "paint", "paints", "the", "and", "for", "with", "a", "an", "of", "in", "on",
    "interior", "exterior", "acrylic", "water", "based", "oil", "enamel", "tds", "sds",
    "data", "sheet", "technical", "safety", "product", "range", "new", "formula",
}


def norm(s: str) -> str:
    s = (s or "").lower().replace("’", "'").replace("&", " and ")
    s = re.sub(r"[®™©]", "", s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = s.replace("'", "")
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in norm(s).split() if t not in GENERIC and len(t) > 1}


def detect_filters(question: str, products: list[str], brands: list[str],
                   min_token_share: float = 0.75) -> dict:
    """Return {"product": [..], "brand": [..]} (lists may be empty).

    Products: exact normalised phrase match wins, longest first (so the more
    specific product name beats the shorter one it contains). Otherwise the
    product whose distinctive tokens are best covered by the question wins, if
    coverage >= min_token_share ("Wash&Wear Low Sheen" -> "Wash&Wear 101 Low Sheen").
    Ambiguous questions (two products equally matched below the bar) get no filter.
    """
    q = f" {norm(question)} "
    q_tokens = set(q.split())

    # strip brand words from product names for matching ("Dulux Aquanamel" ~ "Aquanamel")
    brand_tokens = {t for b in brands for t in norm(b).split()}

    exact = [p for p in products if norm(p) and f" {norm(p)} " in q]
    if exact:
        longest = max(len(norm(p)) for p in exact)
        # keep every exact match that is not a strict substring of a longer match
        hits = [p for p in exact if len(norm(p)) == longest
                or not any(norm(p) in norm(o) and p != o for o in exact)]
    else:
        scored = []
        for p in products:
            toks = _tokens(p) - brand_tokens
            if not toks:
                continue
            share = len(toks & q_tokens) / len(toks)
            if share >= min_token_share:
                scored.append((share, len(toks), p))
        best = max(((sh, n) for sh, n, _ in scored), default=None)
        hits = [p for sh, n, p in scored if (sh, n) == best]

    brand_hits = [b for b in brands if b and f" {norm(b)} " in q]
    return {"product": sorted(set(hits)), "brand": sorted(set(brand_hits))}
