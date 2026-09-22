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


def duspec_style(path):
    """Like a DuSpec+ sheet: Title Case headings with no colon, inline 'Uses: ...' headings,
    and a table in the MIDDLE of the page between two headings."""
    doc = SimpleDocTemplate(path, pagesize=A4)
    body = ss["BodyText"]
    tbl = Table([["Touch dry", "45 minutes"], ["Recoat", "3 hours"], ["Coverage", "12 m2 per litre"]],
                colWidths=[150, 250])
    tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story = [
        Paragraph("Acme TrimCoat Satin", body),
        Paragraph("Introduction", body),
        Paragraph("Part A 999 LINE", body),
        Paragraph("Description and Image", body),
        Paragraph("Acme TrimCoat Satin is a water based enamel for interior doors and trim that dries to a hard satin finish.", body),
        Paragraph("Uses: Use Acme TrimCoat Satin on doors, windows, skirting boards and architraves where a durable washable finish is required.", body),
        Paragraph("Application", body),
        tbl, Spacer(1, 8),
        Paragraph("Clean Up", body),
        Paragraph("Clean brushes and rollers in water immediately after use. Do not pour leftover paint down the drain.", body),
    ]
    doc.build(story)


def two_column(path):
    """Like a Selleys sheet: two text columns, a table in the right column, full-width footer."""
    from reportlab.platypus import BaseDocTemplate, Frame, FrameBreak, PageTemplate
    W, H = A4
    frames = [Frame(40, 80, (W - 100) / 2, H - 160, id="left"),
              Frame(60 + (W - 100) / 2, 80, (W - 100) / 2, H - 160, id="right")]

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(40, 40, "Head Office 1 Example Street Sydney NSW 2000 T: 1300 000 000 E: service@brightco.example")

    doc = BaseDocTemplate(path, pagesize=A4)
    doc.addPageTemplates([PageTemplate(id="two", frames=frames, onPage=footer)])
    body = ss["BodyText"]
    tbl = Table([["Property", "Typical Result"], ["Open Time", "20 minutes"], ["Cure time", "24 hours"]],
                colWidths=[110, 110])
    tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story = [
        Paragraph("GripFix Construction Adhesive", ss["Heading2"]),
        Paragraph("Description", ss["Heading3"]),
        Paragraph("GripFix is a high strength multipurpose construction adhesive that forms a strong and lasting bond on most building materials including timber and plasterboard.", body),
        Paragraph("Uses", ss["Heading3"]),
        Paragraph("GripFix is suitable for timber, plasterboard, MDF, masonry, concrete, tiles and metals in interior and exterior applications.", body),
        Paragraph("Approvals and Standards", ss["Heading3"]),
        Paragraph("GripFix meets the requirements of the example construction adhesive standard for wet and dry timber.", body),
        FrameBreak(),
        Paragraph("Technical Details", ss["Heading3"]),
        tbl, Spacer(1, 8),
        Paragraph("How To Use", ss["Heading3"]),
        Paragraph("Ensure surfaces are free from oil, grease and dust before applying the adhesive in beads every 40 cm along each stud or batten.", body),
        Paragraph("Press pieces firmly together and allow the adhesive to set for 24 hours before removing temporary fasteners.", body),
    ]
    doc.build(story)


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
    duspec_style(f"{out_dir}/acme-trimcoat-satin-tds.pdf")
    two_column(f"{out_dir}/brightco-gripfix-tds.pdf")
    with open(f"{out_dir}/manifest.csv", "w") as f:
        f.write("file,brand,product,doc_type,source_url\n")
        f.write("acme-wallguard-low-sheen-tds.pdf,Acme,WallGuard Low Sheen,TDS,https://example.com/a.pdf\n")
        f.write("acme-wallguard-kitchen-bathroom-tds.pdf,Acme,WallGuard Kitchen & Bathroom,TDS,https://example.com/b.pdf\n")
    return out_dir


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/pdfs"))
