"""Download the public PDFs listed in a manifest CSV (run on your laptop).

  python scripts/download_pdfs.py data/manifest.csv data/pdfs

The manifest needs at least: file, brand, product, doc_type, source_url.
Only use publicly available documents from official sources.
Checks every download is really a PDF (some links return an HTML page instead).
"""
from __future__ import annotations

import csv
import os
import sys
import time

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (docs-rag-lakehouse portfolio project; public data sheets)"}


def main(manifest: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ok, bad = 0, []
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("source_url")]
    for r in rows:
        dest = os.path.join(out_dir, r["file"])
        if os.path.exists(dest):
            ok += 1
            continue
        try:
            resp = requests.get(r["source_url"], headers=HEADERS, timeout=30)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                raise ValueError(f"not a PDF (content-type {resp.headers.get('content-type')})")
            with open(dest, "wb") as out:
                out.write(resp.content)
            ok += 1
            print(f"OK   {r['file']} ({len(resp.content) // 1024} KB)")
        except Exception as e:
            bad.append(r["file"])
            print(f"FAIL {r['file']}: {e}")
        time.sleep(0.5)  # be polite
    print(f"\n{ok}/{len(rows)} PDFs in {out_dir}" + (f"; failed: {bad}" if bad else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/manifest.csv",
         sys.argv[2] if len(sys.argv) > 2 else "data/pdfs")
