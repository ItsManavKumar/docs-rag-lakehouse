import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from docs_rag.chunking import canonical_section, chunk_document, detect_heading  # noqa: E402
from docs_rag.clean import content_hash, normalize_text, repeated_lines, strip_noise_lines  # noqa: E402
from docs_rag.evaluate import retrieval_hit  # noqa: E402
from docs_rag.extract import table_to_lines  # noqa: E402
from docs_rag.filters import detect_filters  # noqa: E402
from docs_rag.index import VectorIndex  # noqa: E402
from docs_rag.metadata import infer_doc_type, infer_product  # noqa: E402
from docs_rag.rag import cited_sources  # noqa: E402


def test_headings():
    assert detect_heading("4. FIRST AID MEASURES") == "Section 4: First Aid Measures"
    assert detect_heading("SECTION 7: HANDLING AND STORAGE") == "Section 7: Handling And Storage"
    assert detect_heading("Surface Preparation") == "Surface Preparation"
    assert detect_heading("RECOAT TIME") == "Recoat Time"
    assert detect_heading("Recoat: 2 hours") is None          # table row, not a heading
    assert detect_heading("1. Apply two coats with a roller") is None  # numbered step
    assert canonical_section("Recoat Time") == "Recoat time"
    assert canonical_section("Section 4: First Aid Measures") == "First aid"


def test_table_rendering():
    assert table_to_lines([["Property", "Value"], ["Recoat", "2 hours"]]) == ["Recoat: 2 hours"]
    wide = table_to_lines([["Size", "Coverage", "Pack"], ["1L", "16 m2", "Tin"]])
    assert wide == ["Size: 1L; Coverage: 16 m2; Pack: Tin"]


def test_cleaning():
    assert normalize_text("sur-\nface") == "surface"
    assert normalize_text("2-\n4 hours") == "2-\n4 hours"      # ranges are not joined
    pages = ["Acme Pty Ltd Page 1 of 2\nbody one", "Acme Pty Ltd Page 2 of 2\nbody two"]
    drop = repeated_lines(pages)
    assert strip_noise_lines(pages[0], drop) == "body one"
    assert content_hash("Keep  dry.") == content_hash("keep dry")


def test_metadata():
    assert infer_doc_type("SAFETY DATA SHEET\n1. IDENTIFICATION", "x.pdf") == "SDS"
    assert infer_product("Acme WallGuard Low Sheen\nTECHNICAL DATA SHEET", "a-tds.pdf", "Acme") == "WallGuard Low Sheen"
    assert infer_product("SAFETY DATA SHEET\nProduct name: Acme Deck Oil", "x.pdf", "Acme") == "Deck Oil"


def test_filter_prefers_most_specific_product():
    products = ["WallGuard Low Sheen", "WallGuard Kitchen & Bathroom", "WallGuard"]
    f = detect_filters("recoat time for wallguard kitchen and bathroom?", products, ["Acme"])
    assert f["product"] == ["WallGuard Kitchen & Bathroom"]
    f = detect_filters("How long before I recoat WallGuard?", products, ["Acme"])
    assert f["product"] == ["WallGuard"]
    assert detect_filters("what is a primer?", products, ["Acme"])["product"] == []
    ww = ["Wash&Wear 101 Low Sheen", "Wash&Wear +Plus Kitchen & Bathroom"]
    assert detect_filters("Recoat time for Dulux Wash & Wear Low Sheen?", ww, ["Dulux"])["product"] == [ww[0]]
    assert detect_filters("wash and wear kitchen bathroom drying", ww, ["Dulux"])["product"] == [ww[1]]
    assert detect_filters("wash and wear recoat time", ww, ["Dulux"])["product"] == []  # ambiguous


def test_index_filter():
    recs = [{"chunk_id": str(i), "product": p, "file": f"{p}.pdf", "page_start": 1, "page_end": 1}
            for i, p in enumerate(["A", "B", "A"])]
    vecs = np.eye(3, dtype="float32")
    idx = VectorIndex(vecs, recs)
    assert [h["chunk_id"] for h in idx.search(vecs[1], k=3, filters={"product": ["A"]})] in (["0", "2"], ["2", "0"])
    assert idx.search(vecs[1], k=1)[0]["chunk_id"] == "1"


def test_chunk_split_and_pages():
    long = " ".join(f"word{i}." for i in range(500))
    pages = [{"page": 1, "text": "APPLICATION\n" + long}, {"page": 2, "text": "more text on page two"}]
    chunks = chunk_document(pages, {"file": "f.pdf", "brand": "B", "product": "P", "doc_type": "TDS"},
                            max_words=100, overlap_words=10, min_words=5)
    assert all(c["n_words"] <= 110 for c in chunks)
    assert chunks[-1]["page_end"] == 2
    assert chunks[0]["embed_text"].startswith("B | P | TDS | Application")


def test_retrieval_hit_and_citations():
    q = {"answerable": True, "expected_file": "a.pdf", "expected_pages": [2]}
    assert retrieval_hit(q, [{"file": "a.pdf", "page_start": 1, "page_end": 2}], 5) is True
    assert retrieval_hit(q, [{"file": "a.pdf", "page_start": 3, "page_end": 3}], 5) is False
    chunks = [{"file": "a.pdf", "page_start": 1, "page_end": 1, "product": "P", "section": "S"}]
    assert cited_sources("Two hours [S1][S9].", chunks)[0]["file"] == "a.pdf"


def test_end_to_end_offline(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from make_fixtures import build
    build(os.path.join(ROOT, "tests", "fixtures", "pdfs"))
    env = {**os.environ, "DOCS_RAG_LAKE": str(tmp_path)}
    run = lambda *a: subprocess.run([sys.executable, "scripts/run_local.py", *a, "--test-mode",
                                     "--config", "tests/test_config.yaml"], cwd=ROOT, env=env,
                                    capture_output=True, text=True, check=True).stdout
    out = run("all")
    assert "duplicate document dropped: acme-wallguard-low-sheen-sds-copy.pdf" in out
    assert "need OCR" in out
    ans = run("ask", "What is the recoat time for Acme WallGuard Kitchen & Bathroom?")
    assert "WallGuard Kitchen & Bathroom" in ans and "metadata filter" in ans
    ev = run("eval")
    assert "retrieval_hit_rate" in ev


# --- layouts found on the real corpus -------------------------------------------------

@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from make_fixtures import build
    return build(str(tmp_path_factory.mktemp("fx")))


def _chunks(fixture_dir, name):
    from docs_rag.extract import extract_pages
    with open(os.path.join(fixture_dir, name), "rb") as f:
        pages = extract_pages(f.read())
    meta = {"file": name, "brand": "B", "product": "P", "doc_type": "TDS"}
    return pages, chunk_document(pages, meta, 220, 40, 12)


def test_duspec_style_headings_and_table_position(fixture_dir):
    pages, chunks = _chunks(fixture_dir, "acme-trimcoat-satin-tds.pdf")
    by_section = {c["section"]: c for c in chunks}
    assert "Uses" in by_section                                  # inline "Uses: ..." heading split out
    assert by_section["Uses"]["text"].startswith("Use Acme TrimCoat")
    assert "Recoat: 3 hours" in by_section["Application"]["text"]  # table stays under its heading
    assert "Recoat" not in by_section["Clean Up"]["text"]
    assert by_section["Clean Up"]["section_canonical"] == "Clean up"


def test_two_column_reading_order(fixture_dir):
    pages, chunks = _chunks(fixture_dir, "brightco-gripfix-tds.pdf")
    assert pages[0]["n_columns"] == 2
    text = pages[0]["text"]
    # left column is read top-to-bottom before the right column; lines are not merged across
    assert text.index("Approvals and Standards") < text.index("Technical Details") < text.index("How To Use")
    assert "Approvals and Standards\n" in text
    assert text.rstrip().endswith("service@brightco.example")      # full-width footer kept whole
    how = next(c for c in chunks if c["section"] == "How To Use")
    assert how["section_canonical"] == "Application" and "24 hours" in how["text"]


def test_single_column_pages_not_split(fixture_dir):
    from docs_rag.extract import extract_pages
    with open(os.path.join(fixture_dir, "acme-wallguard-low-sheen-tds.pdf"), "rb") as f:
        assert all(p["n_columns"] == 1 for p in extract_pages(f.read()))


def test_heading_rules():
    from docs_rag.chunking import split_inline_heading
    assert detect_heading("Description and Image") == "Description and Image"
    assert detect_heading("Wet and dry timber") is None           # not title case
    assert detect_heading("Liquid Nails Original") is None        # no section keyword
    assert split_inline_heading("Uses: Use it on doors and trim where a durable finish is needed.")[0] == "Uses"
    assert split_inline_heading("Recoat: 2 hours") is None         # table row, not a heading
    assert canonical_section("Secure fixing") == "General"         # 'cur' must not match 'secure'
    assert detect_heading("Standards & Certificates") == "Standards & Certificates"
    assert canonical_section("Small Spills") == "Spills"
    assert canonical_section("Large Spills") == "Spills"
    assert canonical_section("Dangerous Good Classification") == "Transport"
    assert canonical_section("Chemical Entity Cas No Proportion") == "Composition"
    assert canonical_section("Introduction") == "Product description"
    assert canonical_section("Approvals & Standards") == "Approvals"
    assert canonical_section("Standards & Certificates") == "Approvals"
    assert canonical_section("Product Information") == "Technical data"
    assert canonical_section("Maintenance") == "Maintenance"
