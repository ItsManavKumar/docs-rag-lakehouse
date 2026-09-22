"""Vector index over the Gold table with optional metadata filtering.

FAISS (exact inner product on normalised vectors = cosine) when installed,
numpy otherwise. A few thousand chunks don't need approximate search.
"""
from __future__ import annotations

import json
import os

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

META_COLS = ("chunk_id", "file", "brand", "product", "doc_type", "section",
             "section_canonical", "page_start", "page_end", "text", "source_url")


class VectorIndex:
    def __init__(self, vectors: np.ndarray, records: list[dict]):
        assert len(vectors) == len(records), "vectors and records must align"
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.records = records
        self.backend = "faiss" if faiss is not None else "numpy"
        self._faiss = None
        if faiss is not None and len(records):
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)

    @classmethod
    def from_rows(cls, rows: list[dict], vector_key: str = "embedding") -> "VectorIndex":
        vecs = np.array([r[vector_key] for r in rows], dtype="float32")
        recs = [{k: r.get(k) for k in META_COLS} for r in rows]
        return cls(vecs, recs)

    def _mask(self, filters: dict | None) -> np.ndarray | None:
        if not filters:
            return None
        mask = np.ones(len(self.records), dtype=bool)
        applied = False
        for key, allowed in filters.items():
            if allowed:
                allowed = set(allowed)
                mask &= np.array([r.get(key) in allowed for r in self.records])
                applied = True
        return mask if applied else None

    def search(self, query_vec: np.ndarray, k: int = 5, filters: dict | None = None) -> list[dict]:
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        mask = self._mask(filters)
        if mask is not None:
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                return []
            scores = (self.vectors[idx] @ q[0])
            order = np.argsort(-scores)[:k]
            pairs = [(int(idx[i]), float(scores[i])) for i in order]
        elif self._faiss is not None:
            s, i = self._faiss.search(q, min(k, len(self.records)))
            pairs = [(int(a), float(b)) for a, b in zip(i[0], s[0]) if a != -1]
        else:
            scores = self.vectors @ q[0]
            order = np.argsort(-scores)[:k]
            pairs = [(int(i), float(scores[i])) for i in order]
        return [{**self.records[i], "score": round(sc, 4), "rank": r + 1}
                for r, (i, sc) in enumerate(pairs)]

    # persistence (FAISS file + JSON metadata) -------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "vectors.npy"), self.vectors)
        with open(os.path.join(directory, "records.json"), "w") as f:
            json.dump(self.records, f)
        if self._faiss is not None:
            faiss.write_index(self._faiss, os.path.join(directory, "index.faiss"))

    @classmethod
    def load(cls, directory: str) -> "VectorIndex":
        vecs = np.load(os.path.join(directory, "vectors.npy"))
        with open(os.path.join(directory, "records.json")) as f:
            recs = json.load(f)
        return cls(vecs, recs)


class DatabricksVSIndex:
    """Same interface as VectorIndex, backed by a Databricks Vector Search index.

    `records` (metadata only, no vectors) is still needed so the Retriever can
    detect product/brand names in questions.
    """

    backend = "databricks_vs"

    def __init__(self, vs_index, records: list[dict]):
        self.vs_index = vs_index
        self.records = records

    def search(self, query_vec, k: int = 5, filters: dict | None = None) -> list[dict]:
        cols = list(META_COLS)
        kwargs = {"query_vector": [float(x) for x in query_vec], "columns": cols, "num_results": k}
        if filters:
            kwargs["filters"] = {key: list(v) for key, v in filters.items() if v}
        res = self.vs_index.similarity_search(**kwargs)
        rows = res.get("result", {}).get("data_array", [])
        names = [c["name"] for c in res.get("manifest", {}).get("columns", [])]
        out = []
        for rank, row in enumerate(rows, start=1):
            rec = dict(zip(names, row))
            score = rec.pop("score", None)
            out.append({**{c: rec.get(c) for c in cols}, "score": score, "rank": rank})
        return out
