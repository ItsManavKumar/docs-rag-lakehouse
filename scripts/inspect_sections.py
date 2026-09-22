"""How well did section detection work? Run after `python scripts/run_local.py all`.

  python scripts/inspect_sections.py            # summary
  python scripts/inspect_sections.py --samples  # also show the start of 'General' chunks per brand

Reports the share of chunks with no recognised section ("General"), per brand and per file,
plus which pages were read as two-column. Use it to compare before/after a chunking change.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAKE = os.environ.get("DOCS_RAG_LAKE", os.path.join(ROOT, "data", "local_lake"))

b = pd.read_parquet(os.path.join(LAKE, "bronze_pages.parquet"))
s = pd.read_parquet(os.path.join(LAKE, "silver_chunks.parquet"))
s["is_general"] = s.section_canonical.eq("General")

total = len(s)
gen = int(s.is_general.sum())
print(f"chunks: {total}   'General' (no recognised section): {gen} ({gen / total:.0%})")
print("\n=== General share by brand ===")
print(s.groupby("brand").is_general.agg(chunks="size", general="sum", share="mean").round(2))
print("\n=== canonical sections ===")
print(s.section_canonical.value_counts().to_string())
print("\n=== worst files (General share) ===")
print(s.groupby("file").is_general.mean().sort_values(ascending=False).head(10).round(2).to_string())
if "n_columns" in b.columns:
    two = b[b.n_columns == 2]
    print(f"\npages read as two-column: {len(two)} of {len(b)}  ({sorted(two.file.unique())[:8]})")

if "--samples" in sys.argv:
    for brand, g in s[s.is_general].groupby("brand"):
        print(f"\n=== sample 'General' chunks: {brand} ===")
        for _, r in g.head(4).iterrows():
            print(f"- [{r.file} p.{r.page_start} | heading: {r.section}]\n  {r.text[:220]!r}")
