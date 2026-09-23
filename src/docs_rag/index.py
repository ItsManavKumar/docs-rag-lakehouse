"""Search over the Gold table: vectors, keywords, or both.

- Vector search: FAISS (exact inner product on normalised vectors = cosine) when
  installed, numpy otherwise. A few thousand chunks don't need approximate search.
- Keyword search: a small BM25 implementation (no extra dependency).
- Hybrid: both lists merged with Reciprocal Rank Fusion.

Why hybrid: vector search finds passages that *mean* the same thing, but it dilutes a
rare exact phrase sitting inside a long noisy passage. A real miss on this corpus: the
recoat time for 1 Step Prep is in a chunk titled "Clean Up", buried among film
thicknesses and spread rates, and ranked 9th for "What is the recoat time...?". BM25
finds that chunk instantly because it contains the literal words "Recoat Time".
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

# without this, "what is the ... for" dominates BM25 and short chunks win on stopwords alone
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "if", "in", "is", "it", "long", "me", "much", "my", "of", "on", "or", "should", "that",
    "the", "their", "there", "this", "to", "use", "used", "using", "was", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your",
}


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if not drop_stopwords or t not in STOPWORDS]


class BM25:
    """Okapi BM25 over the chunk texts. Small corpus, so a plain inverted index is fine."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = [0] * len(docs)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            tokens = tokenize(doc)
            self.doc_len[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        self.n = len(docs)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def search(self, query: str, k: int = 20, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for i, tf in self.postings[term]:
                if allowed is not None and i not in allowed:
                    continue
                dl = self.doc_len[i] or 1
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]


def reciprocal_rank_fusion(rankings: list[list[int]], rrf_k: int = 60) -> list[int]:
    """Merge ranked id lists: each list contributes 1/(rrf_k + rank) to an id's score.

    Rank-based, so the two searches' incomparable score scales don't matter.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] += 1.0 / (rrf_k + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


META_COLS = ("chunk_id", "file", "brand", "product", "doc_type", "section",
             "section_canonical", "page_start", "page_end", "text", "source_url")


class VectorIndex:
    def __init__(self, vectors: np.ndarray, records: list[dict]):
        assert len(vectors) == len(records), "vectors and records must align"
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.records = records
        self.backend = "faiss" if faiss is not None else "numpy"
        self._chunk_pos = {r["chunk_id"]: i for i, r in enumerate(records)}
        self._faiss = None
        self._bm25: BM25 | None = None
        if faiss is not None and len(records):
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)

    @classmethod
    def from_rows(cls, rows: list[dict], vector_key: str = "embedding") -> "VectorIndex":
        vecs = np.array([r[vector_key] for r in rows], dtype="float32")
        recs = [{k: r.get(k) for k in META_COLS} for r in rows]
        return cls(vecs, recs)

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            # section + text only: the product name is the filter dimension, and including
            # it would make every chunk of a product match a question naming that product
            self._bm25 = BM25([f"{r.get('section', '')} {r.get('text', '')}" for r in self.records])
        return self._bm25

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

    def search_hybrid(self, query_vec: np.ndarray, query_text: str, k: int = 5,
                      filters: dict | None = None, pool: int = 20) -> list[dict]:
        """Vector search + BM25 keyword search, merged with Reciprocal Rank Fusion."""
        mask = self._mask(filters)
        allowed = set(np.flatnonzero(mask).tolist()) if mask is not None else None
        if allowed is not None and not allowed:
            return []
        dense_hits = self.search(query_vec, pool, filters)
        dense_ids = [self._id_of(h) for h in dense_hits]
        lexical_ids = [i for i, _ in self.bm25.search(query_text, pool, allowed)]
        merged = reciprocal_rank_fusion([dense_ids, lexical_ids])[:k]
        dense_scores = {self._id_of(h): h["score"] for h in dense_hits}
        out = []
        for rank, i in enumerate(merged, start=1):
            rec = self.records[i]
            out.append({**rec, "rank": rank, "score": dense_scores.get(i),
                        "found_by": ("both" if i in dense_scores and i in set(lexical_ids)
                                     else "vector" if i in dense_scores else "keyword")})
        return out

    def _id_of(self, hit: dict) -> int:
        return self._chunk_pos[hit["chunk_id"]]

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
