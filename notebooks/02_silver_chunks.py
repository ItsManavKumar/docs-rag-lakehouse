# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: clean, de-duplicate, chunk by section ("AI-ready data")
# MAGIC
# MAGIC | Step | How |
# MAGIC |---|---|
# MAGIC | Clean | Unicode/ligature normalisation, re-join hyphenated words, strip repeated headers/footers and "Page x of y" |
# MAGIC | Chunk | Section-aware: SDS numbered sections, known TDS headings (Recoat time, Surface preparation, ...), split only if > `max_words` |
# MAGIC | Tag | brand, product, doc type, section, canonical section, page range |
# MAGIC | De-dup (documents) | same normalised text under a different file name → keep one (window function) |
# MAGIC | De-dup (chunks) | identical chunk text **within the same product** → keep one |
# MAGIC | Data quality | checks table written to `silver_dq_checks` |

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

from pyspark.sql import functions as F

from docs_rag.metadata import load_manifest
from docs_rag.spark_jobs import build_silver, chunks_on_driver, dedupe_chunks, dq_checks

bronze = spark.table(T["bronze"]).filter(F.col("corpus") == CORPUS)
manifest_files = list(load_manifest(cfg["corpus"]["manifest"]))

try:
    silver, dropped = build_silver(bronze, cfg, manifest_files, SRC_DIR)
    silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T["silver"])
    print("Chunked on Spark workers (applyInPandas)")
except Exception as e:
    # If the workers can't import docs_rag from the workspace files, chunk on the driver instead.
    print("Worker-side chunking failed, falling back to driver:", repr(e)[:200])
    silver, dropped = dedupe_chunks(chunks_on_driver(spark, bronze, cfg), manifest_files)
    silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T["silver"])

spark.sql(f"COMMENT ON TABLE {T['silver']} IS 'Silver: cleaned, de-duplicated, section-aware chunks tagged with product/brand/section/page'")

# COMMAND ----------

# MAGIC %md ### Duplicate documents removed

# COMMAND ----------

display(dropped)

# COMMAND ----------

# MAGIC %md ### Data quality checks

# COMMAND ----------

silver = spark.table(T["silver"])
dq = dq_checks(bronze, silver, cfg["chunking"]["max_words"]).withColumn("checked_at", F.current_timestamp())
dq.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_dq_checks")
display(dq)

# COMMAND ----------

# MAGIC %md ### What the chunks look like

# COMMAND ----------

display(silver.groupBy("section_canonical").agg(F.count("*").alias("chunks"),
                                                 F.round(F.avg("n_words")).alias("avg_words"))
              .orderBy(F.desc("chunks")))

# COMMAND ----------

display(silver.select("product", "doc_type", "section", "page_start", "page_end", "n_words", "embed_text")
              .orderBy("file", "chunk_index").limit(20))
