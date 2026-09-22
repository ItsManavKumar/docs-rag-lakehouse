"""Document-level metadata: brand, product, doc type.

A manifest CSV is the source of truth when present (it also records the public
source URL of every PDF). Otherwise we infer from the filename and first page.
"""
from __future__ import annotations

import csv
import os
import re

SDS_MARKERS = ("safety data sheet", "material safety data sheet")
TDS_MARKERS = ("technical data sheet", "product data sheet", "technical data")


def load_manifest(path: str | None) -> dict[str, dict]:
    """file name -> {brand, product, doc_type, source_url}. Missing file -> {}."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["file"].strip(): {k: (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(f) if row.get("file")}


def infer_doc_type(first_page_text: str, file_name: str) -> str:
    t = (first_page_text[:1500] + " " + file_name).lower()
    if any(m in t for m in SDS_MARKERS) or re.search(r"\bsds\b", t):
        return "SDS"
    if any(m in t for m in TDS_MARKERS) or re.search(r"\btds\b", t):
        return "TDS"
    return "OTHER"


def infer_brand(text: str, file_name: str, known_brands: list[str]) -> str:
    hay = (file_name.replace("_", " ").replace("-", " ") + " " + text[:3000]).lower()
    hay = hay.replace("’", "'")
    best, best_pos = "", 10**9
    for b in known_brands:
        pos = hay.find(b.lower())
        if pos == -1 and "'" in b:
            pos = hay.find(b.lower().replace("'", ""))
        if pos != -1 and pos < best_pos:
            best, best_pos = b, pos
    return best or "Unknown"


_PRODUCT_FIELD = re.compile(
    r"(?:product\s*name|product\s*identifier|trade\s*name)\s*[:\-]?\s*(.+)", re.I)


_NOT_TITLES = ("data sheet", "datasheet", "safety data", "technical data", "product information",
               "issue date", "revision", "page ")


def _strip_brand(name: str, brand: str) -> str:
    if brand and brand != "Unknown":
        for b in (brand, brand.replace("'", "")):
            if name.lower().startswith(b.lower() + " "):
                return name[len(b):].strip()
    return name


def infer_product(first_page_text: str, file_name: str, brand: str = "") -> str:
    # 1) SDS: an explicit "Product name:" field
    m = _PRODUCT_FIELD.search(first_page_text)
    if m:
        name = m.group(1).strip().split("\n")[0]
        name = re.split(r"\s{2,}|\bRecommended use\b|\bSynonyms?\b", name)[0].strip(" :-")
        if 2 < len(name) < 90:
            return _strip_brand(name, brand)
    # 2) TDS: the title is usually the first short line that isn't "Technical Data Sheet"
    for line in first_page_text.split("\n")[:6]:
        t = line.strip()
        if 2 < len(t) < 80 and len(t.split()) <= 9 and not any(x in t.lower() for x in _NOT_TITLES):
            if brand and brand != "Unknown" and brand.lower().replace("'", "") in t.lower().replace("'", ""):
                return _strip_brand(t, brand)
            break
    # 3) fall back to the file name
    stem = os.path.splitext(os.path.basename(file_name))[0]
    stem = re.sub(r"(?i)[_\- ]*(tds|sds|msds|datasheet|data[_\- ]sheet)[_\- ]*", " ", stem)
    return _strip_brand(re.sub(r"[_\-]+", " ", stem).strip().title(), brand)


def doc_metadata(file_name: str, first_page_text: str, manifest: dict[str, dict],
                 known_brands: list[str]) -> dict:
    base = os.path.basename(file_name)
    m = manifest.get(base, {})
    brand = m.get("brand") or infer_brand(first_page_text, base, known_brands)
    return {
        "brand": brand,
        "product": m.get("product") or infer_product(first_page_text, base, brand),
        "doc_type": m.get("doc_type") or infer_doc_type(first_page_text, base),
        "source_url": m.get("source_url", ""),
    }
