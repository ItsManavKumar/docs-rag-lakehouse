"""Load config/config.yaml and derive table names."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    p = cfg["project"]
    ns = f"{p['catalog']}.{p['schema']}"
    cfg["tables"] = {
        "bronze": f"{ns}.bronze_pages",
        "silver": f"{ns}.silver_chunks",
        "gold": f"{ns}.gold_chunk_embeddings",
        "eval": f"{ns}.eval_results",
        "qa_log": f"{ns}.qa_log",
    }
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg
