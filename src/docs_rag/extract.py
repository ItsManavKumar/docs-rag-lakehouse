"""PDF -> per-page text (Bronze layer).

Three layout problems found on the real data sheets, and how this module handles them:

1. Tables. Plain text extraction reads tables row-by-row across columns and mixes
   them with nearby text ("Touch Dry 30 Recoat 2 minutes hours"). We find tables with
   pdfplumber, extract the text outside them, and render each table row as
   "Property: value".
   Boxes that only look like tables (a border around the page or a text panel) are
   rejected, otherwise the whole page would be squashed into one line.
2. Table position. Tables are placed back where they sit on the page (sorted by their
   top edge), so a drying-time table stays under the heading above it instead of
   being appended at the end of the page.
3. Two-column pages. Some data sheets (e.g. Selleys) use two text columns; reading
   straight across merges unrelated sentences ("Approvals & Standards with skin and
   eyes."). We detect a vertical gutter and read the left column, then the right,
   band by band, so full-width headers, footers and tables stay in order.
"""
from __future__ import annotations

import io
import logging
import re

import pdfplumber

try:  # fallback extractor
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

# pdfminer logs a warning for every malformed font descriptor; they are harmless
logging.getLogger("pdfminer").setLevel(logging.ERROR)

_GENERIC_HEADER = {"property", "properties", "parameter", "item", "value", "values", "details",
                   "characteristic", "typical value", "typical result", "result", "test", "units"}


def _clean_cell(c) -> str:
    return re.sub(r"\s+", " ", str(c)).strip() if c is not None else ""


def table_to_lines(table: list[list]) -> list[str]:
    """Render a table as readable 'key: value' lines.

    - empty columns are dropped first (merged cells often leave blank columns)
    - 2-column tables (the common data-sheet layout: property | value) -> "property: value"
    - wider tables with a header row -> "header1: v1; header2: v2" per row
    """
    rows = [[_clean_cell(c) for c in row] for row in table if row]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width) if any(r[i] for r in rows)]
    rows = [[r[i] for i in keep] for r in rows]
    width = len(keep)
    lines: list[str] = []
    if width == 1:
        return [r[0] for r in rows]
    if width == 2:
        if all(c.lower().rstrip("*") in _GENERIC_HEADER for c in rows[0] if c):
            rows = rows[1:]  # a "Property | Value" header row adds nothing
        for k, v in rows:
            if k and v:
                lines.append(f"{k}: {v}")
            elif k or v:
                lines.append(k or v)
        return lines
    header = rows[0]
    has_header = sum(bool(h) for h in header) >= width - 1 and len(rows) > 1
    body = rows[1:] if has_header else rows
    for r in body:
        cells = []
        for i, c in enumerate(r):
            if not c:
                continue
            h = header[i] if has_header else ""
            cells.append(f"{h}: {c}" if h else c)
        if cells:
            lines.append("; ".join(cells))
    return lines


# ---------------------------------------------------------------- columns ---

def _rows_of_words(words: list[dict], tol: float = 3.0) -> list[list[dict]]:
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if rows and abs(rows[-1][0]["top"] - w["top"]) <= tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    return rows


# a column gutter is a wide gap; wide letter spacing inside a word is a narrow one.
# "W e a t h e r s h i e l d" in a letter-spaced SDS title was being split mid-word.
MIN_GUTTER = 14.0


def _crosses(row: list[dict], x: float, gap: float = MIN_GUTTER) -> bool:
    """Does this line of text run through x (i.e. it is not split by a wide gap at x)?"""
    for w in row:
        if w["x0"] < x < w["x1"]:
            return True
    left = [w["x1"] for w in row if w["x1"] <= x]
    right = [w["x0"] for w in row if w["x0"] >= x]
    return bool(left and right and min(right) - max(left) < gap)


def _layout_words(page, table_bboxes: list[tuple]) -> list[dict]:
    """Words outside tables, plus one block per table, so a wide 2-column table
    (property | value) can't be mistaken for two text columns."""
    def inside(w):
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in table_bboxes)
    words = [w for w in page.extract_words() if not inside(w)]
    words += [{"x0": b[0], "x1": b[2], "top": b[1], "bottom": b[3], "text": "<table>"} for b in table_bboxes]
    return words


def find_gutter(page, words: list[dict] | None = None) -> float | None:
    """x position of a two-column gutter, or None for a single-column page.

    Two columns = most lines stop short of the gutter or start after it (few lines run
    through it), and both sides carry a real share of the lines.
    """
    words = page.extract_words() if words is None else words
    if len(words) < 40:
        return None
    rows = _rows_of_words(words)
    n = len(rows)
    candidates = []
    for frac in (i / 100 for i in range(38, 63)):
        x = page.width * frac
        crossing = [_crosses(r, x) for r in rows]
        left = sum(1 for r, c in zip(rows, crossing) if not c and any(w["x1"] <= x for w in r))
        right = sum(1 for r, c in zip(rows, crossing) if not c and any(w["x0"] >= x for w in r))
        # a right-aligned page number or date must not make a page "two-column"
        if (sum(crossing) <= 0.3 * n and min(left, right) >= max(4, 0.25 * n)):
            candidates.append((sum(crossing), x))
    if not candidates:
        return None
    best = min(c for c, _ in candidates)
    xs = [x for c, x in candidates if c == best]
    return xs[len(xs) // 2]  # middle of the gutter


def _bands(page, x: float, words: list[dict]) -> list[tuple[float, float, bool]]:
    """Split the page into horizontal bands: (top, bottom, is_full_width)."""
    rows = _rows_of_words(words)
    bands: list[list] = []
    for r in rows:
        full = _crosses(r, x)
        top, bottom = min(w["top"] for w in r), max(w["bottom"] for w in r)
        if bands and bands[-1][2] == full:
            bands[-1][1] = max(bands[-1][1], bottom)
        else:
            bands.append([top, bottom, full])
    # stretch bands to cover the gaps between them so nothing falls through
    out = []
    for i, (top, bottom, full) in enumerate(bands):
        t = page.bbox[1] if i == 0 else (bands[i - 1][1] + top) / 2
        b = page.bbox[3] if i == len(bands) - 1 else (bottom + bands[i + 1][0]) / 2
        out.append((t, b, full))
    return out


# ------------------------------------------------------------ extraction ---

def is_real_table(table, region) -> bool:
    """Reject 'tables' that are really boxes.

    Some data sheets (e.g. DuSpec+) draw a border around the page or around a text panel.
    pdfplumber sees the rectangle as a one-cell table, and rendering it as a table would
    squash the whole page into one line, hiding the headings inside it. A real table has
    at least 2 rows and 2 columns of content and doesn't cover most of the page.
    """
    rows = [[c for c in r if c is not None and str(c).strip()] for r in (table.extract() or [])]
    rows = [r for r in rows if r]
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return False
    x0, top, x1, bottom = table.bbox
    rx0, rtop, rx1, rbottom = region.bbox
    return (x1 - x0) * (bottom - top) <= 0.7 * (rx1 - rx0) * (rbottom - rtop)


def find_real_tables(region) -> list:
    return [t for t in region.find_tables() if is_real_table(t, region)]


def _region_text(region, tables_as_key_value: bool) -> tuple[str, int]:
    """Text of a page region with tables rendered in place (ordered by vertical position)."""
    tables = find_real_tables(region) if tables_as_key_value else []
    bboxes = [t.bbox for t in tables]

    def outside_tables(obj) -> bool:
        if obj.get("object_type") != "char":
            return True
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        return not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in bboxes)

    body = region.filter(outside_tables) if bboxes else region
    items = [(ln["top"], ln["text"]) for ln in body.extract_text_lines() if ln["text"].strip()]
    for t in tables:
        lines = table_to_lines(t.extract())
        if lines:
            items.append((t.bbox[1], "\n".join(lines)))
    items.sort(key=lambda it: it[0])
    return "\n".join(text for _, text in items), len(tables)


def page_text(page, tables_as_key_value: bool = True) -> tuple[str, int, int]:
    """Return (text, n_tables, n_columns) for one page."""
    table_bboxes = [t.bbox for t in find_real_tables(page)] if tables_as_key_value else []
    words = _layout_words(page, table_bboxes)
    x = find_gutter(page, words)
    if x is None:
        text, n = _region_text(page, tables_as_key_value)
        return text, n, 1
    parts, n_tables = [], 0
    x0, _, x1, _ = page.bbox
    for top, bottom, full in _bands(page, x, words):
        if bottom - top < 1:
            continue
        crops = ([(x0, top, x1, bottom)] if full else [(x0, top, x, bottom), (x, top, x1, bottom)])
        for bbox in crops:
            text, n = _region_text(page.crop(bbox), tables_as_key_value)
            n_tables += n
            if text.strip():
                parts.append(text)
    return "\n".join(parts), n_tables, 2


def extract_pages(pdf_bytes: bytes, tables_as_key_value: bool = True) -> list[dict]:
    """One dict per page: page (1-based), text, n_tables, n_columns, extractor, needs_ocr."""
    pages: list[dict] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text, n_tables, n_cols = page_text(page, tables_as_key_value)
                    extractor = "pdfplumber"
                except Exception:  # a single malformed page shouldn't kill the document
                    text, n_tables, n_cols, extractor = page.extract_text() or "", 0, 1, "pdfplumber-plain"
                pages.append({"page": i, "text": text, "n_tables": n_tables, "n_columns": n_cols,
                              "extractor": extractor})
    except Exception:
        if PdfReader is None:
            raise
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [{"page": i, "text": p.extract_text() or "", "n_tables": 0, "n_columns": 1,
                  "extractor": "pypdf"} for i, p in enumerate(reader.pages, start=1)]
    for p in pages:
        # Scanned/image-only pages come back empty; flag them rather than silently dropping.
        p["needs_ocr"] = len(p["text"].strip()) < 20
    return pages
