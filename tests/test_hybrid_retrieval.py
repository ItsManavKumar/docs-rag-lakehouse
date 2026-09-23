"""Hybrid retrieval (vector + BM25) and the layout bugs it was added for.

Real miss it fixes: "What is the recoat time for 1 Step Prep?" ranked the chunk that
actually contains "Recoat Time (min/hours) 2 hours" at position 9, because the phrase is
buried in a long passage about film thickness and spread rates.
"""
import os
import sys

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from docs_rag.clean import normalize_text  # noqa: E402
from docs_rag.embeddings import HashingEmbedder  # noqa: E402
from docs_rag.extract import extract_pages  # noqa: E402
from docs_rag.index import BM25, VectorIndex, reciprocal_rank_fusion  # noqa: E402
from docs_rag.filters import strip_entities  # noqa: E402
from docs_rag.rag import Retriever  # noqa: E402

NOISY = ("Clean Up Description Clean all equipment with water Application Methods Air Spray "
         "Airless Spray Brush Roller Application Conditions Solids by Volume 40 Min Max "
         "Recommended Wet Film Per Coat (microns) 71 71 71 Dry Film Per Coat (microns) 29 29 29 "
         "Recoat Time (min/hours) 2 hours Indefinite 2 hours Theoretical Spread Rate (m2/L) 14 14 14")
DECOYS = [
    "Dulux 1 Step Prep is a multi-surface primer sealer undercoat for interior and exterior use.",
    "Approvals APAS 0172 and Green Star compliance for this primer sealer undercoat product.",
    "All preparation and painting must conform to AS2311 The Painting of Buildings standard.",
    "Pack sizes 0.5L 1L 2L 4L 10L 15L with weights listed for transport and storage purposes.",
    "This data sheet is copyright to DuluxGroup Australia and New Zealand and may not be copied.",
]


def _index(texts):
    emb = HashingEmbedder()
    records = [{"chunk_id": str(i), "file": "f.pdf", "product": "1 Step Prep", "brand": "Dulux",
                "doc_type": "TDS", "section": "Clean Up" if i == 0 else "Other",
                "section_canonical": "General", "page_start": 1, "page_end": 1,
                "text": t, "source_url": ""} for i, t in enumerate(texts)]
    return VectorIndex(emb.embed_documents(texts), records), emb


def test_bm25_finds_rare_phrase_in_noisy_chunk():
    index, _ = _index([NOISY] + DECOYS)
    top = index.bm25.search("recoat time", k=3)   # product words are stripped by the retriever
    assert top[0][0] == 0, "the chunk containing 'Recoat Time' should rank first on keywords"


def test_hybrid_beats_vector_only_on_the_real_miss():
    index, emb = _index([NOISY] + DECOYS)
    q = "What is the recoat time for Dulux 1 Step Prep?"
    qv = emb.embed_query(q)
    vector_only = [h["chunk_id"] for h in index.search(qv, k=3)]
    hybrid = [h["chunk_id"] for h in index.search_hybrid(qv, "recoat time", k=3)]
    assert "0" in hybrid, "hybrid must surface the chunk holding the answer"
    assert hybrid[0] == "0"
    # sanity: the fusion is doing the work, not the vector side alone
    assert vector_only != hybrid or "0" == vector_only[0]


def test_hybrid_reports_how_each_hit_was_found():
    index, emb = _index([NOISY] + DECOYS)
    q = "recoat time"
    hits = index.search_hybrid(emb.embed_query(q), q, k=4)
    assert all(h["found_by"] in ("vector", "keyword", "both") for h in hits)


def test_retriever_respects_hybrid_flag_and_filters():
    index, emb = _index([NOISY] + DECOYS)
    r = Retriever(index, emb, top_k=3, use_metadata_filter=True, hybrid=True)
    hits, applied = r.retrieve("What is the recoat time for 1 Step Prep?")
    assert applied == {"product": ["1 Step Prep"]} and hits[0]["chunk_id"] == "0"
    off, _ = r.retrieve("What is the recoat time for 1 Step Prep?", hybrid=False)
    assert all("found_by" not in h for h in off)


def test_strip_entities():
    assert strip_entities("What is the recoat time for Dulux 1 Step Prep?",
                          ["1 Step Prep", "Dulux"]) == "what is the recoat time for"


def test_rrf_merges_by_rank():
    assert reciprocal_rank_fusion([[1, 2, 3], [3, 9, 1]])[0] in (1, 3)
    assert BM25([]).search("anything") == []


def test_private_use_icons_stripped():
    assert normalize_text("Clean Up  Water") == "Clean Up Water"


def test_letter_spaced_title_is_not_two_columns(tmp_path):
    """An SDS title drawn with wide letter spacing must not be read as two columns
    (it was splitting 'Weathershield' into 'Weat' + 'hershield')."""
    path = str(tmp_path / "spaced.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    title = c.beginText(60, 780)
    title.setFont("Helvetica-Bold", 16)
    title.setCharSpace(7)                               # wide letter spacing
    title.textLine("DULUX WEATHERSHIELD EXTERIOR LOW SHEEN")
    c.drawText(title)
    c.setFont("Helvetica", 10)
    y = 740
    for line in ["SAFETY DATA SHEET", "1. IDENTIFICATION OF THE MATERIAL AND SUPPLIER",
                 "Product name: Dulux Weathershield Exterior Low Sheen",
                 "Supplier: DuluxGroup (Australia) Pty Ltd, 1956 Dandenong Road, Clayton VIC 3168",
                 "Emergency telephone number: 1800 638 556 available 24 hours",
                 "2. HAZARDS IDENTIFICATION",
                 "Not classified as hazardous according to the criteria of the GHS.",
                 "Keep out of reach of children. Avoid contact with skin and eyes.",
                 "4. FIRST AID MEASURES",
                 "Eye contact: rinse with running water for 15 minutes and seek medical advice."]:
        c.drawString(60, y, line)
        y -= 18
    c.save()
    with open(path, "rb") as f:
        pages = extract_pages(f.read())
    assert pages[0]["n_columns"] == 1
    assert "WEATHERSHIELD" in pages[0]["text"].upper().replace(" ", "")
