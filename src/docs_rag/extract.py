"""PDF -> per-page text (Bronze layer).

Why not just `page.extract_text()`?
Technical Data Sheets put the most important facts (drying time, recoat time,
coverage) in tables. Plain text extraction reads tables row-by-row across
columns and often interleaves them with the text beside them, producing lines
like "Touch Dry 30 Recoat 2 minutes hours". So we:
  1. find tables with pdfplumber,
  2. extract the page text *outside* the table boxes,
  3. append each table as "Header: value" lines.
"""
from __future__ import annotations

import io
import re

import pdfplumber

try:  # fallback extractor
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


def _clean_cell(c) -> str:
    return re.sub(r"\s+", " ", str(c)).strip() if c is not None else ""


def table_to_lines(table: list[list]) -> list[str]:
    """Render a table as readable 'key: value' lines.

    - 2-column tables (the common TDS layout: property | value) -> "property: value"
    - wider tables with a header row -> "header1: v1; header2: v2" per row
    """
    rows = [[_clean_cell(c) for c in row] for row in table if row]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    lines: list[str] = []
    if width == 2:
        generic = {"property", "properties", "parameter", "item", "value", "values",
                   "details", "characteristic", "typical value", "result"}
        if all(c.lower() in generic for c in rows[0] if c):
            rows = rows[1:]  # a "Property | Value" header row adds nothing
        for r in rows:
            k, v = (r + ["", ""])[:2]
            if k and v:
                lines.append(f"{k}: {v}")
            elif k or v:
                lines.append(k or v)
        return lines
    header = rows[0]
    has_header = all(header) and len(rows) > 1
    body = rows[1:] if has_header else rows
    for r in body:
        cells = []
        for i, c in enumerate(r):
            if not c:
                continue
            h = header[i] if has_header and i < len(header) else ""
            cells.append(f"{h}: {c}" if h else c)
        if cells:
            lines.append("; ".join(cells))
    return lines


def _page_text_with_tables(page, tables_as_key_value: bool) -> tuple[str, int]:
    if not tables_as_key_value:
        return page.extract_text() or "", 0
    found = page.find_tables()
    if not found:
        return page.extract_text() or "", 0
    bboxes = [t.bbox for t in found]

    def outside_tables(obj) -> bool:
        if obj.get("object_type") != "char":
            return True
        x = (obj["x0"] + obj["x1"]) / 2
        y = (obj["top"] + obj["bottom"]) / 2
        return not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in bboxes)

    text = page.filter(outside_tables).extract_text() or ""
    table_lines: list[str] = []
    for t in found:
        table_lines.extend(table_to_lines(t.extract()))
    if table_lines:
        text = text.rstrip() + "\n" + "\n".join(table_lines)
    return text, len(found)


def extract_pages(pdf_bytes: bytes, tables_as_key_value: bool = True) -> list[dict]:
    """Return one dict per page: page (1-based), text, n_tables, extractor, needs_ocr."""
    pages: list[dict] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text, n_tables = _page_text_with_tables(page, tables_as_key_value)
                    extractor = "pdfplumber"
                except Exception:  # a single malformed page shouldn't kill the doc
                    text, n_tables, extractor = page.extract_text() or "", 0, "pdfplumber-plain"
                pages.append({"page": i, "text": text, "n_tables": n_tables,
                              "extractor": extractor})
    except Exception:
        if PdfReader is None:
            raise
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [{"page": i, "text": p.extract_text() or "", "n_tables": 0, "extractor": "pypdf"}
                 for i, p in enumerate(reader.pages, start=1)]
    for p in pages:
        # Scanned/image-only pages come back empty; flag them rather than silently dropping.
        p["needs_ocr"] = len(p["text"].strip()) < 20
    return pages
