"""Draft evaluation questions from your chunks, for YOU to verify against the PDFs.

  python scripts/run_local.py all          # builds data/local_lake/silver_chunks.parquet
  python scripts/draft_eval_questions.py   # -> eval/questions_draft.jsonl

Samples chunks across products and sections and asks the LLM for one factual
question per chunk with its answer. Every draft MUST be checked by opening the
PDF at the stated page before it goes into eval/questions.jsonl; a question set
written by the same model that answers it would flatter the results.
Unanswerable questions are left for you to write.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from docs_rag.config import load_config  # noqa: E402
from docs_rag.llm import build_client  # noqa: E402

PROMPT = """From the data sheet excerpt below, write ONE specific factual question a painter or customer
might ask, whose answer is stated explicitly in the excerpt. Name the product in the question.
Reply with JSON only: {"question": "...", "expected_answer": "...", "expected_keywords": ["..."]}"""

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
cfg = load_config()
lake = os.environ.get("DOCS_RAG_LAKE", os.path.join(ROOT, "data", "local_lake"))
s = pd.read_parquet(os.path.join(lake, "silver_chunks.parquet"))
s = s[s.n_words >= 25]
useful = ["Recoat time", "Drying time", "Coverage", "Surface preparation", "Application", "Technical data",
          "First aid", "Handling and storage", "Clean up", "Exposure controls / PPE"]
pool = s[s.section_canonical.isin(useful)]
sample = pool.groupby("product", group_keys=False).apply(lambda g: g.sample(min(2, len(g)), random_state=7))
sample = sample.sample(min(N, len(sample)), random_state=7)

llm = build_client(cfg)
out_path = os.path.join(ROOT, "eval", "questions_draft.jsonl")
with open(out_path, "w") as f:
    for i, row in enumerate(sample.itertuples(), start=1):
        raw = llm.chat([{"role": "system", "content": PROMPT},
                        {"role": "user", "content": f"Product: {row.brand} {row.product}\n\n{row.text}"}])
        try:
            d = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except ValueError:
            continue
        f.write(json.dumps({"id": f"q{i:02d}", **d, "answerable": True, "expected_file": row.file,
                            "expected_pages": list(range(row.page_start, row.page_end + 1)),
                            "_verify": "OPEN THE PDF AND CHECK, then delete this field"}) + "\n")
print("wrote", out_path)
