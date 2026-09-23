# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Retrieval + grounded answers with citations
# MAGIC
# MAGIC 1. Embed the question (bge query prefix) and retrieve the top-k chunks.
# MAGIC    If the question names a product (or brand) we know, search **only that product's chunks**.
# MAGIC 2. Send the chunks, numbered `[S1]..[Sk]` with file, page and section, to Claude Sonnet 5
# MAGIC    through the Vercel AI Gateway (OpenAI-compatible API).
# MAGIC 3. The model must answer only from the sources, cite `[Sn]` after each fact, and reply
# MAGIC    *"That information is not in the documents."* when it can't.
# MAGIC 4. Every Q&A is logged to the `qa_log` Delta table (question, answer, citations, retrieved chunks).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import json

from pyspark.sql import functions as F

from docs_rag.embeddings import build_embedder
from docs_rag.index import META_COLS, DatabricksVSIndex, VectorIndex
from docs_rag.llm import build_client
from docs_rag.rag import Retriever, answer_question, pretty

r = cfg["retrieval"]
if r["vector_backend"] == "databricks_vs":
    from databricks.vector_search.client import VectorSearchClient
    vs = VectorSearchClient(disable_notice=True).get_index(r["vs_endpoint"], f"{CATALOG}.{SCHEMA}.gold_chunk_index")
    meta = [row.asDict() for row in spark.table(T["gold"]).select(*META_COLS).collect()]
    index = DatabricksVSIndex(vs, meta)
else:
    index = VectorIndex.load(r["index_dir"])

retriever = Retriever(index, build_embedder(cfg), r["top_k"], r["use_metadata_filter"], r.get("hybrid", True))
llm = build_client(cfg, dbutils=dbutils)
print(f"{len(index.records)} chunks, backend={index.backend}, {len(retriever.products)} products, model={cfg['llm']['model']}")

# COMMAND ----------

# MAGIC %md ### Ask a question
# MAGIC Replace these with questions about the documents you loaded (product names must match your corpus).

# COMMAND ----------

questions = [
    "What is the recoat time for <PRODUCT NAME>?",
    "How should I prepare bare plaster before painting with <PRODUCT NAME>?",
    "What first aid is recommended for eye contact with <PRODUCT NAME>?",
    "What does <PRODUCT NAME> cost per litre?",       # should be 'not in the documents'
]
results = [answer_question(q, retriever, llm) for q in questions]
for res in results:
    print(pretty(res), "\n")

# COMMAND ----------

# MAGIC %md ### Why did it answer that? Inspect the retrieved chunks

# COMMAND ----------

import pandas as pd
display(pd.DataFrame([{**c, "question": res["question"]} for res in results for c in res["retrieved"]]))

# COMMAND ----------

# MAGIC %md ### Filter on vs. off (the similar-product-names problem)

# COMMAND ----------

q = questions[0]
for use_filter in (False, True):
    hits, applied = retriever.retrieve(q, use_filter=use_filter)
    print(f"filter={use_filter} {applied}")
    for h in hits:
        print(f"   {h['rank']}. {h['score']:.3f}  {h['product']:<40} {h['section']:<30} p.{h['page_start']}")

# COMMAND ----------

# MAGIC %md ### Log the Q&A to Delta

# COMMAND ----------

log = spark.createDataFrame([{
    "question": x["question"], "answer": x["answer"], "not_found": x["not_found"],
    "citations": json.dumps(x["citations"]), "filters_applied": json.dumps(x["filters_applied"]),
    "retrieved": json.dumps(x["retrieved"]), "model": cfg["llm"]["model"], "corpus": CORPUS,
} for x in results]).withColumn("asked_at", F.current_timestamp())
log.write.mode("append").option("mergeSchema", "true").saveAsTable(T["qa_log"])
display(spark.table(T["qa_log"]).orderBy(F.desc("asked_at")).limit(10))
