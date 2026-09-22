# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze: raw PDFs → page text
# MAGIC
# MAGIC Reads every PDF in the corpus Volume with Spark's `binaryFile` source and parses them in
# MAGIC parallel with `mapInPandas`. One row per page, with document metadata
# MAGIC (brand, product, doc type, public source URL) from `manifest.csv` or inferred from the text.
# MAGIC
# MAGIC Tables in PDFs are rendered as `key: value` lines (see `src/docs_rag/extract.py`).
# MAGIC Pages with no text layer are **flagged** (`needs_ocr`), not silently dropped.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

from pyspark.sql import functions as F

from docs_rag.metadata import load_manifest
from docs_rag.spark_jobs import BRONZE_SCHEMA, build_bronze, read_pdfs

manifest = load_manifest(cfg["corpus"]["manifest"])
raw = read_pdfs(spark, cfg["corpus"]["pdf_dir"])
n_pdfs = raw.count()
print(f"{n_pdfs} PDFs found, {len(manifest)} manifest entries")
assert n_pdfs > 0, f"No PDFs in {cfg['corpus']['pdf_dir']}"

# COMMAND ----------

try:
    bronze = build_bronze(raw, cfg, manifest, SRC_DIR)
    bronze.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T["bronze"])
    print("Parsed on Spark workers (mapInPandas)")
except Exception as e:
    # If the workers can't import docs_rag from the workspace files, parse on the driver instead.
    # 30-50 PDFs is small enough for this to take seconds.
    print("Worker-side parsing failed, falling back to driver:", repr(e)[:200])
    import pandas as pd
    from docs_rag.pipeline import BRONZE_COLUMNS, bronze_rows_for_pdf
    rows = []
    for r in raw.select("path", "content").toLocalIterator():
        rows.extend(bronze_rows_for_pdf(r.path, bytes(r.content), cfg, manifest))
    bronze = (spark.createDataFrame(pd.DataFrame(rows, columns=BRONZE_COLUMNS), BRONZE_SCHEMA)
              .withColumn("ingested_at", F.current_timestamp()))
    bronze.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T["bronze"])

spark.sql(f"COMMENT ON TABLE {T['bronze']} IS 'Bronze: one row per PDF page, raw extracted text + document metadata'")

# COMMAND ----------

b = spark.table(T["bronze"])
display(b.groupBy("brand", "doc_type").agg(F.countDistinct("file").alias("files"), F.count("*").alias("pages"),
                                           F.sum("n_tables").alias("tables_parsed"),
                                           F.sum(F.col("needs_ocr").cast("int")).alias("pages_need_ocr"))
          .orderBy("brand", "doc_type"))

# COMMAND ----------

# MAGIC %md Spot-check a page with a table: the table rows should read as `Property: value`.

# COMMAND ----------

display(b.filter("n_tables > 0").select("file", "product", "page", "text").limit(3))
