"""Stage functions shared by the Databricks notebooks and scripts/run_local.py.

Each function works on plain Python rows so it can run inside a Spark
mapInPandas / applyInPandas UDF or on a laptop with pandas.
"""
from __future__ import annotations

import hashlib
import os

from .chunking import chunk_document
from .clean import content_hash, normalize_text, repeated_lines, strip_noise_lines
from .extract import extract_pages
from .metadata import doc_metadata

BRONZE_COLUMNS = ["corpus", "file", "path", "file_sha256", "brand", "product", "doc_type",
                  "source_url", "page", "n_pages", "text", "n_chars", "n_tables", "extractor",
                  "needs_ocr"]

SILVER_COLUMNS = ["corpus", "chunk_id", "chunk_index", "file", "brand", "product", "doc_type",
                  "source_url", "section", "section_canonical", "page_start", "page_end", "text",
                  "embed_text", "n_words", "content_hash"]


def bronze_rows_for_pdf(path: str, pdf_bytes: bytes, cfg: dict, manifest: dict) -> list[dict]:
    file_name = os.path.basename(path)
    pages = extract_pages(pdf_bytes, cfg["extraction"]["tables_as_key_value"])
    first_text = "\n".join(p["text"] for p in pages[:2])
    meta = doc_metadata(file_name, first_text, manifest, cfg["corpus"]["known_brands"])
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    return [{
        "corpus": cfg["corpus"]["name"], "file": file_name, "path": path, "file_sha256": sha,
        **meta, "page": p["page"], "n_pages": len(pages), "text": p["text"],
        "n_chars": len(p["text"]), "n_tables": p["n_tables"], "extractor": p["extractor"],
        "needs_ocr": p["needs_ocr"],
    } for p in pages]


def silver_chunks_for_doc(pages: list[dict], cfg: dict) -> list[dict]:
    """pages: all Bronze rows for ONE document."""
    if not pages:
        return []
    pages = sorted(pages, key=lambda r: r["page"])
    texts = [normalize_text(p["text"]) for p in pages]
    if cfg["extraction"].get("strip_repeated_lines", True):
        drop = repeated_lines(texts)
        texts = [strip_noise_lines(t, drop) for t in texts]
    first = pages[0]
    meta = {k: first[k] for k in ("file", "brand", "product", "doc_type")}
    ch = cfg["chunking"]
    chunks = chunk_document([{"page": p["page"], "text": t} for p, t in zip(pages, texts)], meta,
                            ch["max_words"], ch["overlap_words"], ch["min_words"])
    for c in chunks:
        c["corpus"] = first["corpus"]
        c["source_url"] = first.get("source_url", "")
        c["content_hash"] = content_hash(c["text"])
    return chunks


def doc_text_hash(pages: list[dict]) -> str:
    """Hash of a document's normalised full text (catches re-uploads under new names)."""
    return content_hash("".join(normalize_text(p["text"]) for p in sorted(pages, key=lambda r: r["page"])))
