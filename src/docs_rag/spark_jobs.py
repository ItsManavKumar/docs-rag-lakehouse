"""PySpark transformations for the Bronze and Silver layers.

Notebooks do the I/O (read Volume, write Delta); these functions do the
transformations, so they can be unit-tested with a local SparkSession.

PDF parsing and chunking run inside mapInPandas / applyInPandas. The UDFs add
`src_dir` to sys.path on the worker so `docs_rag` is importable there too.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

BRONZE_SCHEMA = T.StructType([
    T.StructField("corpus", T.StringType()), T.StructField("file", T.StringType()),
    T.StructField("path", T.StringType()), T.StructField("file_sha256", T.StringType()),
    T.StructField("brand", T.StringType()), T.StructField("product", T.StringType()),
    T.StructField("doc_type", T.StringType()), T.StructField("source_url", T.StringType()),
    T.StructField("page", T.IntegerType()), T.StructField("n_pages", T.IntegerType()),
    T.StructField("text", T.StringType()), T.StructField("n_chars", T.IntegerType()),
    T.StructField("n_tables", T.IntegerType()), T.StructField("n_columns", T.IntegerType()),
    T.StructField("extractor", T.StringType()),
    T.StructField("needs_ocr", T.BooleanType()),
])

SILVER_SCHEMA = T.StructType([
    T.StructField("corpus", T.StringType()), T.StructField("chunk_id", T.StringType()),
    T.StructField("chunk_index", T.IntegerType()), T.StructField("file", T.StringType()),
    T.StructField("brand", T.StringType()), T.StructField("product", T.StringType()),
    T.StructField("doc_type", T.StringType()), T.StructField("source_url", T.StringType()),
    T.StructField("section", T.StringType()), T.StructField("section_canonical", T.StringType()),
    T.StructField("page_start", T.IntegerType()), T.StructField("page_end", T.IntegerType()),
    T.StructField("text", T.StringType()), T.StructField("embed_text", T.StringType()),
    T.StructField("n_words", T.IntegerType()), T.StructField("content_hash", T.StringType()),
    T.StructField("doc_hash", T.StringType()),
])


def _bronze_pdf_fn(cfg: dict, manifest: dict, src_dir: str):
    def fn(batches):
        import sys

        import pandas as pd
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from docs_rag.pipeline import BRONZE_COLUMNS, bronze_rows_for_pdf

        for pdf in batches:
            rows = []
            for path, content in zip(pdf["path"], pdf["content"]):
                rows.extend(bronze_rows_for_pdf(path, bytes(content), cfg, manifest))
            yield pd.DataFrame(rows, columns=BRONZE_COLUMNS)
    return fn


def read_pdfs(spark: SparkSession, pdf_dir: str) -> DataFrame:
    return (spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.pdf")
            .option("recursiveFileLookup", "true")
            .load(pdf_dir)
            .select("path", "content", "length", "modificationTime"))


def build_bronze(raw: DataFrame, cfg: dict, manifest: dict, src_dir: str) -> DataFrame:
    """One row per PDF page. Parsing is distributed across files."""
    return (raw.select("path", "content")
            .repartition(max(1, min(64, raw.count())))  # spread PDFs over workers
            .mapInPandas(_bronze_pdf_fn(cfg, manifest, src_dir), BRONZE_SCHEMA)
            .withColumn("ingested_at", F.current_timestamp()))


def _silver_doc_fn(cfg: dict, src_dir: str):
    def fn(pdf):
        import sys

        import pandas as pd
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from docs_rag.pipeline import doc_text_hash, silver_chunks_for_doc

        pages = pdf.to_dict("records")
        chunks = silver_chunks_for_doc(pages, cfg)
        h = doc_text_hash(pages)
        for c in chunks:
            c["doc_hash"] = h
        cols = [f.name for f in SILVER_SCHEMA.fields]
        return pd.DataFrame(chunks, columns=cols)
    return fn


def dedupe_chunks(chunks: DataFrame, manifest_files: list[str]) -> tuple[DataFrame, DataFrame]:
    """Document-level de-dup (window over doc_hash) + chunk-level de-dup within a product.

    Returns (silver, dropped_docs). When the same content appears under two file
    names we keep the one listed in the manifest, then the shortest name
    ("x.pdf" beats "x-copy.pdf"). Identical chunk text is only collapsed within
    the same brand/product/doc type: boilerplate shared by different products
    stays, so a product-filtered search can still find it.
    """
    in_manifest = F.col("file").isin(manifest_files) if manifest_files else F.lit(False)
    docs = chunks.select("file", "doc_hash").distinct()
    w = Window.partitionBy("doc_hash").orderBy((~in_manifest).cast("int"), F.length("file"), "file")
    ranked = docs.withColumn("rn", F.row_number().over(w)) \
                 .withColumn("kept_file", F.first("file").over(w))
    keep = ranked.filter("rn = 1").select("file")
    dropped = ranked.filter("rn > 1").select(F.col("file").alias("dropped_file"), "kept_file", "doc_hash")
    silver = (chunks.join(keep, "file", "inner")
              .dropDuplicates(["brand", "product", "doc_type", "content_hash"])
              .withColumn("processed_at", F.current_timestamp()))
    return silver, dropped


def build_silver(bronze: DataFrame, cfg: dict, manifest_files: list[str], src_dir: str) -> tuple[DataFrame, DataFrame]:
    """Returns (silver_chunks, dropped_duplicate_docs).

    1. drop pages with no text layer (reported by the DQ checks)
    2. chunk each document (section-aware)  -> applyInPandas per file
    3. de-duplicate documents and chunks     -> dedupe_chunks
    """
    pages = bronze.filter(~F.col("needs_ocr")).drop("ingested_at")
    # (no .cache(): serverless compute doesn't support it; the data is small)
    chunks = pages.groupBy("file").applyInPandas(_silver_doc_fn(cfg, src_dir), SILVER_SCHEMA)
    return dedupe_chunks(chunks, manifest_files)


def chunks_on_driver(spark: SparkSession, bronze: DataFrame, cfg: dict) -> DataFrame:
    """Fallback when workers can't import docs_rag: chunk on the driver, same output schema."""
    import pandas as pd

    from .pipeline import doc_text_hash, silver_chunks_for_doc

    pdf = bronze.filter(~F.col("needs_ocr")).drop("ingested_at").toPandas()
    rows = []
    for _, g in pdf.groupby("file"):
        pages = g.to_dict("records")
        h = doc_text_hash(pages)
        rows += [{**c, "doc_hash": h} for c in silver_chunks_for_doc(pages, cfg)]
    cols = [f.name for f in SILVER_SCHEMA.fields]
    return spark.createDataFrame(pd.DataFrame(rows, columns=cols), SILVER_SCHEMA)


def dq_checks(bronze: DataFrame, silver: DataFrame, max_words: int) -> DataFrame:
    """Simple data-quality report: one row per check with pass/fail and a count."""
    spark = bronze.sparkSession
    n_files = bronze.select("file").distinct().count()
    ocr_files = [r.file for r in bronze.filter("needs_ocr").select("file").distinct().collect()]
    files_with_chunks = silver.select("file").distinct().count()
    unknown_brand = silver.filter("brand = 'Unknown' OR brand IS NULL").select("file").distinct().count()
    empty = silver.filter("n_words = 0 OR text IS NULL").count()
    too_long = silver.filter(F.col("n_words") > max_words + 60).count()
    dup_ids = silver.groupBy("chunk_id").count().filter("count > 1").count()
    other_type = silver.filter("doc_type = 'OTHER'").select("file").distinct().count()
    rows = [
        ("files ingested", True, n_files, ""),
        ("pages without a text layer (need OCR)", len(ocr_files) == 0, len(ocr_files), ", ".join(ocr_files)[:500]),
        ("files that produced chunks", files_with_chunks > 0, files_with_chunks, ""),
        ("files with unknown brand", unknown_brand == 0, unknown_brand, "add them to manifest.csv"),
        ("files with unknown doc type", other_type == 0, other_type, "add doc_type to manifest.csv"),
        ("empty chunks", empty == 0, empty, ""),
        ("chunks over the word limit", too_long == 0, too_long, ""),
        ("duplicate chunk_ids", dup_ids == 0, dup_ids, ""),
    ]
    schema = "check STRING, passed BOOLEAN, n LONG, note STRING"
    return spark.createDataFrame(rows, schema)
