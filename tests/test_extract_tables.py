"""Boxed page layouts must not be treated as tables (found on DuSpec+ data sheets).

DuSpec+ pages draw a frame with horizontal dividers between panels. pdfplumber reads
that as a one-column table covering the page, and rendering it as a table squashed each
panel into one line: "Typical Properties Gloss Level Thinner Low Sheen ... Recoat Time 2 hours".
"""
import os
import sys

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from docs_rag.chunking import chunk_document  # noqa: E402
from docs_rag.extract import extract_pages  # noqa: E402

PANELS = [
    ["Precautions and Limitations",
     "Do not apply below 10 degrees C or when rain is expected within two hours."],
    ["Typical Properties", "Touch Dry          30 minutes", "Recoat Time        2 hours",
     "Spread Rate        14 m2 per litre"],
    ["Clean Up", "Clean all equipment with water immediately after use."],
]


def _boxed_page(path):
    W, H = A4
    c = canvas.Canvas(path, pagesize=A4)
    c.setFont("Helvetica", 10)
    top, left, right = H - 60, 40, W - 40
    y = top
    for panel in PANELS:
        c.line(left, y, right, y)                       # divider above each panel
        y -= 18
        for text in panel:
            c.drawString(left + 10, y, text)
            y -= 16
        y -= 8
    c.line(left, y, right, y)
    c.line(left, top, left, y)                          # frame sides
    c.line(right, top, right, y)
    c.setFont("Helvetica", 7)
    c.drawString(left, 30, "Version 11.0 of Datasheet AUXX00001 Page 1 of 1")
    c.save()


def test_boxed_panels_are_not_a_table(tmp_path):
    path = str(tmp_path / "boxed.pdf")
    _boxed_page(path)
    with open(path, "rb") as f:
        pages = extract_pages(f.read())
    text = pages[0]["text"]
    lines = text.split("\n")
    assert pages[0]["n_tables"] == 0                    # the frame is not a table
    assert "Typical Properties" in lines                # heading kept on its own line
    assert any(ln.startswith("Recoat Time") and "2 hours" in ln for ln in lines)
    chunks = chunk_document(pages, {"file": "b.pdf", "brand": "B", "product": "P", "doc_type": "TDS"},
                            220, 40, 3)
    props = next(c for c in chunks if c["section"] == "Typical Properties")
    assert "2 hours" in props["text"] and "Clean all equipment" not in props["text"]
