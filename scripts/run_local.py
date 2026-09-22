"""Run the whole pipeline on a laptop (no Spark), using the same code as the notebooks.

Useful for: developing chunking rules quickly, running the answer step from your
Mac if Databricks Free Edition blocks the LLM endpoint, and CI tests.

  python scripts/run_local.py all                 # bronze -> silver -> gold
  python scripts/run_local.py ask "What is the recoat time for ...?"
  python scripts/run_local.py eval                # 20-question evaluation
  python scripts/run_local.py all --test-mode     # offline: fake embedder + fake LLM

Outputs go to data/local_lake/ as Parquet (mirrors the Delta tables).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from docs_rag.config import load_config  # noqa: E402
from docs_rag.embeddings import build_embedder  # noqa: E402
from docs_rag.evaluate import load_questions, markdown_table, run_eval, summarize  # noqa: E402
from docs_rag.index import VectorIndex  # noqa: E402
from docs_rag.llm import build_client  # noqa: E402
from docs_rag.metadata import load_manifest  # noqa: E402
from docs_rag.pipeline import bronze_rows_for_pdf, doc_text_hash, silver_chunks_for_doc  # noqa: E402
from docs_rag.rag import Retriever, answer_question, pretty  # noqa: E402

LAKE = os.environ.get("DOCS_RAG_LAKE", os.path.join(ROOT, "data", "local_lake"))


def p(*parts):
    return os.path.join(ROOT, *parts)


def bronze(cfg):
    pdf_dir = p(cfg["corpus"]["local_pdf_dir"])
    manifest = load_manifest(p(cfg["corpus"]["local_manifest"]))
    rows = []
    files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not files:
        sys.exit(f"No PDFs found in {pdf_dir}")
    for f in files:
        with open(f, "rb") as fh:
            rows.extend(bronze_rows_for_pdf(f, fh.read(), cfg, manifest))
    df = pd.DataFrame(rows)
    os.makedirs(LAKE, exist_ok=True)
    df.to_parquet(os.path.join(LAKE, "bronze_pages.parquet"), index=False)
    print(f"[bronze] {df.file.nunique()} files, {len(df)} pages, "
          f"{int(df.needs_ocr.sum())} pages flagged needs_ocr, {int(df.n_tables.sum())} tables parsed")
    return df


def silver(cfg):
    b = pd.read_parquet(os.path.join(LAKE, "bronze_pages.parquet"))
    docs = {f: g.to_dict("records") for f, g in b.groupby("file")}
    # 1) document-level de-dup: same content under a different file name
    manifest = load_manifest(p(cfg["corpus"]["local_manifest"]))
    # prefer files listed in the manifest, then the shortest name ("x.pdf" over "x-copy.pdf")
    order = sorted(docs, key=lambda f: (f not in manifest, len(f), f))
    ocr = sorted(b[b.needs_ocr].file.unique())
    if ocr:
        print(f"[silver] pages with no text layer (need OCR, skipped): {ocr}")
    seen, keep = {}, []
    for f in order:
        h = doc_text_hash(docs[f])
        if h in seen:
            print(f"[silver] duplicate document dropped: {f} (same content as {seen[h]})")
            continue
        seen[h] = f
        keep.append(f)
    chunks = [c for f in keep for c in silver_chunks_for_doc(docs[f], cfg)]
    df = pd.DataFrame(chunks)
    # 2) chunk-level de-dup, only within the same product document set
    before = len(df)
    df = df.drop_duplicates(subset=["brand", "product", "doc_type", "content_hash"])
    df.to_parquet(os.path.join(LAKE, "silver_chunks.parquet"), index=False)
    print(f"[silver] {len(df)} chunks ({before - len(df)} duplicate chunks removed); "
          f"median {int(df.n_words.median())} words; sections: "
          f"{df.section_canonical.value_counts().head(8).to_dict()}")
    return df


def gold(cfg, test_mode=False):
    s = pd.read_parquet(os.path.join(LAKE, "silver_chunks.parquet"))
    emb = build_embedder(cfg, test_mode)
    vecs = emb.embed_documents(s.embed_text.tolist())
    s = s.assign(embedding=list(vecs), embedding_model=emb.name)
    s.to_parquet(os.path.join(LAKE, "gold_chunk_embeddings.parquet"), index=False)
    idx = VectorIndex.from_rows(s.to_dict("records"))
    idx.save(os.path.join(LAKE, "faiss"))
    print(f"[gold] {len(s)} vectors, dim={vecs.shape[1]}, model={emb.name}, backend={idx.backend}")


def load_retriever(cfg, test_mode=False):
    idx = VectorIndex.load(os.path.join(LAKE, "faiss"))
    r = cfg["retrieval"]
    return Retriever(idx, build_embedder(cfg, test_mode), r["top_k"], r["use_metadata_filter"])


def ask(cfg, question, test_mode=False):
    res = answer_question(question, load_retriever(cfg, test_mode), build_client(cfg, test_mode=test_mode))
    print(pretty(res))
    return res


def evaluate(cfg, test_mode=False, questions_path=None):
    qs = load_questions(questions_path or p(cfg["evaluation"]["questions_file"]))
    retr = load_retriever(cfg, test_mode)
    llm = build_client(cfg, test_mode=test_mode)
    all_rows, summaries = [], []
    for k, filt in [(5, False), (3, True), (5, True)]:
        rows = run_eval(qs, retr, llm, k=k, use_filter=filt, run_name=f"k{k}_{'filter' if filt else 'nofilter'}",
                        judge_model=cfg["evaluation"]["judge_model"])
        all_rows += rows
        summaries.append(summarize(rows))
    pd.DataFrame(all_rows).to_csv(os.path.join(LAKE, "eval_results.csv"), index=False)
    table = markdown_table(summaries)
    with open(os.path.join(LAKE, "eval_summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bronze", "silver", "gold", "all", "ask", "eval"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--config")
    ap.add_argument("--questions")
    ap.add_argument("--test-mode", action="store_true", help="offline fake embedder + LLM (for tests only)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.stage in ("bronze", "all"):
        bronze(cfg)
    if a.stage in ("silver", "all"):
        silver(cfg)
    if a.stage in ("gold", "all"):
        gold(cfg, a.test_mode)
    if a.stage == "ask":
        ask(cfg, a.question, a.test_mode)
    if a.stage == "eval":
        evaluate(cfg, a.test_mode, a.questions)


if __name__ == "__main__":
    main()
