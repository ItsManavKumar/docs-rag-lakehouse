# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Evaluation
# MAGIC
# MAGIC `eval/questions.jsonl` holds questions written by hand from the PDFs, each with the expected answer,
# MAGIC file and page. Some are deliberately **unanswerable** from the documents.
# MAGIC
# MAGIC | Metric | Meaning |
# MAGIC |---|---|
# MAGIC | retrieval_hit_rate | answerable questions where a chunk from the expected file/page is in the top-k |
# MAGIC | answer_accuracy | all questions judged correct (LLM judge vs. reference answer; unanswerable = must say "not in the documents") |
# MAGIC | unanswerable_refusal_rate | unanswerable questions correctly refused |
# MAGIC | citation_accuracy | answerable questions whose citations include the expected file |
# MAGIC
# MAGIC Three configurations are compared: no metadata filter (k=5), filter (k=3), filter (k=5).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

from docs_rag.embeddings import build_embedder
from docs_rag.evaluate import EVAL_DDL, load_questions, markdown_table, run_eval, summarize
from docs_rag.index import VectorIndex
from docs_rag.llm import build_client
from docs_rag.rag import Retriever

qs = load_questions(os.path.join(REPO_ROOT, cfg["evaluation"]["questions_file"]))
print(f"{len(qs)} questions ({sum(not q['answerable'] for q in qs)} unanswerable)")
assert not any("<" in q["question"] for q in qs), "Replace the template questions in eval/questions.jsonl first"

r = cfg["retrieval"]
retriever = Retriever(VectorIndex.load(r["index_dir"]), build_embedder(cfg), r["top_k"], r["use_metadata_filter"])
llm = build_client(cfg, dbutils=dbutils)

# COMMAND ----------

all_rows, summaries = [], []
for k, use_filter in [(5, False), (3, True), (5, True)]:
    name = f"k{k}_{'filter' if use_filter else 'nofilter'}"
    rows = run_eval(qs, retriever, llm, k=k, use_filter=use_filter, run_name=name,
                    judge_model=cfg["evaluation"]["judge_model"])
    all_rows += rows
    summaries.append(summarize(rows))
    print(name, summaries[-1])

# COMMAND ----------

res = spark.createDataFrame(all_rows, EVAL_DDL).withColumn("evaluated_at", F.current_timestamp())
res.write.mode("append").option("mergeSchema", "true").saveAsTable(T["eval"])
display(spark.createDataFrame(pd.DataFrame(summaries)))

# COMMAND ----------

# MAGIC %md ### Results table for the README

# COMMAND ----------

print(markdown_table(summaries))

# COMMAND ----------

# MAGIC %md ### Failures: read these, they are your "problems I hit" stories

# COMMAND ----------

display(res.filter("run = 'k5_filter' AND (answer_correct = false OR retrieval_hit = false)")
           .select("id", "question", "retrieval_hit", "answer_correct", "judge_reason", "answer", "top_files"))
