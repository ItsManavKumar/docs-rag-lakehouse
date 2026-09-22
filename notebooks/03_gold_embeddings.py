# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold: embeddings + vector index
# MAGIC
# MAGIC - Embeds each Silver chunk's `embed_text` (chunk text prefixed with *brand | product | doc type | section*)
# MAGIC   with `BAAI/bge-small-en-v1.5` (384 dims, runs on CPU).
# MAGIC - Writes vectors to a Delta table (`gold_chunk_embeddings`) with Change Data Feed on, so it can also back a
# MAGIC   Databricks Vector Search index.
# MAGIC - Builds a FAISS index and saves it to a Volume (default backend).
# MAGIC
# MAGIC Embedding runs on the driver: a few thousand chunks take well under a minute, and loading the
# MAGIC model once is cheaper than loading it in every Spark task.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import time

from pyspark.sql import functions as F
from pyspark.sql import types as TT

from docs_rag.embeddings import build_embedder
from docs_rag.index import VectorIndex

silver_pdf = spark.table(T["silver"]).filter(F.col("corpus") == CORPUS).drop("processed_at").toPandas()
emb = build_embedder(cfg)
t0 = time.time()
vecs = emb.embed_documents(silver_pdf["embed_text"].tolist())
print(f"Embedded {len(vecs)} chunks in {time.time() - t0:.1f}s, dim={vecs.shape[1]}, model={emb.name}")

# COMMAND ----------

silver_pdf["embedding"] = [v.tolist() for v in vecs]
silver_pdf["embedding_model"] = emb.name
gold = (spark.createDataFrame(silver_pdf)
        .withColumn("embedding", F.col("embedding").cast(TT.ArrayType(TT.FloatType())))
        .withColumn("embedded_at", F.current_timestamp()))
(gold.write.mode("overwrite").option("overwriteSchema", "true")
     .option("delta.enableChangeDataFeed", "true")
     .saveAsTable(T["gold"]))
spark.sql(f"ALTER TABLE {T['gold']} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
spark.sql(f"COMMENT ON TABLE {T['gold']} IS 'Gold: chunk embeddings ({emb.name}) + metadata for retrieval'")
print("rows:", spark.table(T["gold"]).count())

# COMMAND ----------

# MAGIC %md ### FAISS index (default backend)

# COMMAND ----------

index = VectorIndex.from_rows(silver_pdf.to_dict("records"))
index.save(cfg["retrieval"]["index_dir"])
print(f"Saved {len(index.records)} vectors ({index.backend}) to {cfg['retrieval']['index_dir']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: Databricks Vector Search
# MAGIC Set `retrieval.vector_backend: databricks_vs` in config.yaml to try it. Free Edition allows a single
# MAGIC Vector Search endpoint; if creation fails, stay on FAISS (at this corpus size, exact FAISS search is as good as or better than approximate search).

# COMMAND ----------

if cfg["retrieval"]["vector_backend"] == "databricks_vs":
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(disable_notice=True)
    ep = cfg["retrieval"]["vs_endpoint"]
    if ep not in [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]:
        vsc.create_endpoint_and_wait(name=ep, endpoint_type="STANDARD")
    vsc.create_delta_sync_index_and_wait(
        endpoint_name=ep,
        index_name=f"{CATALOG}.{SCHEMA}.gold_chunk_index",
        source_table_name=T["gold"],
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_dimension=int(vecs.shape[1]),
        embedding_vector_column="embedding",
    )
    print("Vector Search index ready")
else:
    print("vector_backend = faiss; skipping Databricks Vector Search")
