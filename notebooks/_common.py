# Databricks notebook source
# MAGIC %md
# MAGIC ### Shared setup (run by every notebook via `%run ./_common`)
# MAGIC Puts `src/` on the path, loads `config/config.yaml`, and defines paths and table names.

# COMMAND ----------

import os
import sys

NOTEBOOK_DIR = os.getcwd()                       # .../docs-rag-lakehouse/notebooks
REPO_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from docs_rag.config import load_config

cfg = load_config(os.path.join(REPO_ROOT, "config", "config.yaml"))
CATALOG, SCHEMA = cfg["project"]["catalog"], cfg["project"]["schema"]
T = cfg["tables"]
CORPUS = cfg["corpus"]["name"]

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"repo: {REPO_ROOT}\ncorpus: {CORPUS}\ntables: {T}")
