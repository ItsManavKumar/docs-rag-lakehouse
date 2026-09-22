# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup and environment checks
# MAGIC
# MAGIC Run this once. It:
# MAGIC 1. installs the Python libraries,
# MAGIC 2. creates the schema and the Volumes (`raw_pdfs`, `artifacts`, `models`),
# MAGIC 3. checks the three things that can break on Databricks Free Edition:
# MAGIC    - can we download the embedding model from Hugging Face?
# MAGIC    - can we reach the LLM endpoint (Vercel AI Gateway)?
# MAGIC    - is the API key available as a secret?
# MAGIC
# MAGIC Free Edition only allows outbound traffic to a list of trusted domains, so the checks tell you
# MAGIC which fallback (if any) you need. See the README section **"If Free Edition blocks a domain"**.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

for vol in ("raw_pdfs", "artifacts", "models"):
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{vol}")
os.makedirs(cfg["corpus"]["pdf_dir"], exist_ok=True)
print("Upload your PDFs (and manifest.csv) to:", cfg["corpus"]["pdf_dir"])
print("Catalog Explorer > workspace > docs_rag > raw_pdfs > Upload to this volume")

# COMMAND ----------

# MAGIC %md ### Check 1 · Embedding model download

# COMMAND ----------

import requests

def reachable(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        return f"OK (HTTP {r.status_code})"
    except Exception as e:
        return f"BLOCKED ({type(e).__name__})"

print("huggingface.co      ->", reachable("https://huggingface.co/api/models/BAAI/bge-small-en-v1.5"))
print("ai-gateway.vercel.sh ->", reachable("https://ai-gateway.vercel.sh/v1/models"))

# COMMAND ----------

from docs_rag.embeddings import build_embedder

try:
    emb = build_embedder(cfg)
    v = emb.embed_query("recoat time")
    print(f"Embedding model OK: {emb.name}, dim={len(v)}")
except Exception as e:
    print("Could not load the embedding model:", repr(e)[:300])
    print("Fallback: run scripts/download_model.py on your laptop, upload the folder to "
          f"/Volumes/{CATALOG}/{SCHEMA}/models/, and set embedding.model_path in config.yaml")

# COMMAND ----------

# MAGIC %md ### Check 2 · API key + LLM call
# MAGIC Create the secret once from your laptop with the Databricks CLI (the key never goes in a notebook):
# MAGIC ```
# MAGIC databricks secrets create-scope docs-rag
# MAGIC databricks secrets put-secret docs-rag ai-gateway-key
# MAGIC ```

# COMMAND ----------

from docs_rag.llm import build_client

try:
    llm = build_client(cfg, dbutils=dbutils)
    reply = llm.chat([{"role": "user", "content": "Reply with the single word: pong"}])
    print("LLM OK:", cfg["llm"]["model"], "->", reply.strip()[:50])
except Exception as e:
    print("LLM check failed:", repr(e)[:300])
    print("If the gateway domain is blocked, see README > 'If Free Edition blocks a domain'.")
