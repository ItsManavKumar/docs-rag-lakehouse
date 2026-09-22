"""Generate small synthetic data-sheet PDFs for tests.

These are FICTIONAL products from made-up brands ("Acme", "Brightco"). They
exist only to exercise the pipeline offline: tables, numbered SDS sections,
repeated headers/footers, two near-identical product names, a duplicate upload
and a page with no text layer. They are never used for reported results.
"""
from __future__ import annotations

import os
import shutil
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ss = getSampleStyleSheet()


def _footer(brand):
    def draw(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(40, 20, f"{brand} Coatings Pty Ltd - Customer Service 1800 000 000")
        canvas.drawRightString(555, 20, f"Page {doc.page} of 2")
    return draw


def tds(path, brand, product, dry, recoat, coverage, prep, extra_uses):
    doc = SimpleDocTemplate(path, pagesize=A4)
    tbl = Table([["Property", "Value"], ["Finish", "Low sheen"], ["Touch dry", dry],
                 ["Recoat", recoat], ["Coverage", coverage], ["Clean up", "Water"]],
                colWidths=[150, 250])
    tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story = [
        Paragraph(f"{brand} {product}", ss["Title"]),
        Paragraph("TECHNICAL DATA SHEET", ss["Heading2"]),
        Paragraph("PRODUCT DESCRIPTION", ss["Heading3"]),
        Paragraph(f"{product} is a premium water-based acrylic paint for interior walls. {extra_uses}", ss["BodyText"]),
        Paragraph("TECHNICAL DATA", ss["Heading3"]), tbl, Spacer(1, 12),
        PageBreak(),
        Paragraph("SURFACE PREPARATION", ss["Heading3"]),
        Paragraph(prep, ss["BodyText"]),
        Paragraph("APPLICATION", ss["Heading3"]),
        Paragraph("Stir thoroughly before use. Apply two coats by brush, roller or airless spray. "
                  "Do not apply below 10 degrees C or above 35 degrees C.", ss["BodyText"]),
    ]
    doc.build(story, onFirstPage=_footer(brand), onLaterPages=_footer(brand))


def sds(path, brand, product, first_aid):
    doc = SimpleDocTemplate(path, pagesize=A4)
    story = [
        Paragraph("SAFETY DATA SHEET", ss["Title"]),
        Paragraph("1. IDENTIFICATION OF THE MATERIAL AND SUPPLIER", ss["Heading3"]),
        Paragraph(f"Product name: {product}", ss["BodyText"]),
        Paragraph(f"Supplier: {brand} Coatings Pty Ltd", ss["BodyText"]),
        Paragraph("2. HAZARDS IDENTIFICATION", ss["Heading3"]),
        Paragraph("Not classified as hazardous according to the criteria of the GHS.", ss["BodyText"]),
        Paragraph("4. FIRST AID MEASURES", ss["Heading3"]),
        Paragraph(first_aid, ss["BodyText"]),
        PageBreak(),
        Paragraph("7. HANDLING AND STORAGE", ss["Heading3"]),
        Paragraph("Store in a cool, dry place between 5 and 30 degrees C. Protect from frost. Keep container tightly closed.", ss["BodyText"]),
        Paragraph("13. DISPOSAL CONSIDERATIONS", ss["Heading3"]),
        Paragraph("Do not pour leftover paint down the drain. Take to a community recycling centre.", ss["BodyText"]),
    ]
    doc.build(story, onFirstPage=_footer(brand), onLaterPages=_footer(brand))


def blank(path):
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(path, pagesize=A4)
    c.rect(100, 100, 300, 300, fill=1)  # image-like page, no text layer
    c.showPage()
    c.save()


def build(out_dir: str) -> str:
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    tds(f"{out_dir}/acme-wallguard-low-sheen-tds.pdf", "Acme", "WallGuard Low Sheen",
        "30 minutes", "2 hours", "16 m2 per litre",
        "Surfaces must be clean, dry and free of grease. Sand glossy surfaces. Spot prime bare plaster with Acme Sealer.",
        "Suitable for living rooms and bedrooms.")
    tds(f"{out_dir}/acme-wallguard-kitchen-bathroom-tds.pdf", "Acme", "WallGuard Kitchen & Bathroom",
        "1 hour", "4 hours", "14 m2 per litre",
        "Treat mould with a mould killer and rinse. Surfaces must be clean and dry. Spot prime bare plaster.",
        "Contains mould inhibitor for humid areas.")
    tds(f"{out_dir}/brightco-deckcoat-tds.pdf", "Brightco", "DeckCoat Exterior Stain",
        "2 hours", "6 hours", "10 m2 per litre",
        "Remove old flaking coatings. New timber should weather for 4 weeks before coating.",
        "For exterior decks and timber furniture.")
    sds(f"{out_dir}/acme-wallguard-low-sheen-sds.pdf", "Acme", "WallGuard Low Sheen",
        "Eye contact: rinse with running water for 15 minutes. Skin: wash with soap and water.")
    shutil.copy(f"{out_dir}/acme-wallguard-low-sheen-sds.pdf", f"{out_dir}/acme-wallguard-low-sheen-sds-copy.pdf")
    blank(f"{out_dir}/scanned-leaflet.pdf")
    with open(f"{out_dir}/manifest.csv", "w") as f:
        f.write("file,brand,product,doc_type,source_url\n")
        f.write("acme-wallguard-low-sheen-tds.pdf,Acme,WallGuard Low Sheen,TDS,https://example.com/a.pdf\n")
        f.write("acme-wallguard-kitchen-bathroom-tds.pdf,Acme,WallGuard Kitchen & Bathroom,TDS,https://example.com/b.pdf\n")
    return out_dir


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/pdfs"))
