"""Runs the Bronze/Silver Spark transformations on a local SparkSession."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
pyspark = pytest.importorskip("pyspark")

from docs_rag.config import load_config  # noqa: E402
from docs_rag.metadata import load_manifest  # noqa: E402
from docs_rag.spark_jobs import build_bronze, build_silver, dq_checks, read_pdfs  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    s = (SparkSession.builder.master("local[2]").appName("docs-rag-test")
         .config("spark.sql.shuffle.partitions", "4").config("spark.ui.enabled", "false").getOrCreate())
    yield s
    s.stop()


def test_bronze_silver(spark):
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from make_fixtures import build
    pdf_dir = build(os.path.join(ROOT, "tests", "fixtures", "pdfs"))
    cfg = load_config(os.path.join(ROOT, "tests", "test_config.yaml"))
    manifest = load_manifest(os.path.join(pdf_dir, "manifest.csv"))

    bronze = build_bronze(read_pdfs(spark, pdf_dir), cfg, manifest, SRC)
    assert bronze.select("file").distinct().count() == 8
    assert bronze.filter("needs_ocr").count() == 1

    silver, dropped = build_silver(bronze, cfg, list(manifest), SRC)
    d = dropped.collect()
    assert [(r.dropped_file, r.kept_file) for r in d] == [
        ("acme-wallguard-low-sheen-sds-copy.pdf", "acme-wallguard-low-sheen-sds.pdf")]
    assert silver.count() == 24
    assert silver.filter("text LIKE '%Page 1 of%'").count() == 0
    kb = silver.filter("product = 'WallGuard Kitchen & Bathroom' AND section_canonical = 'Technical data'").first()
    assert "Recoat: 4 hours" in kb.text

    dq = {r.check: r for r in dq_checks(bronze, silver, cfg["chunking"]["max_words"]).collect()}
    assert dq["pages without a text layer (need OCR)"].n == 1
    assert dq["duplicate chunk_ids"].passed
