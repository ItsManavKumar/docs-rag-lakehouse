"""Embedding models.

`SentenceTransformerEmbedder` is the real one (bge-small-en-v1.5 by default).
`HashingEmbedder` exists only so tests and CI can run without downloading a
model; never use it for reported results.
"""
from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, model_path: str | None = None,
                 query_prefix: str = "", batch_size: int = 32, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_path or model_name
        self.model = SentenceTransformer(self.name, device=device)
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 200).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0].astype("float32")


class HashingEmbedder:
    """Deterministic bag-of-words hashing vectors. TEST USE ONLY."""

    name = "hashing-test-embedder"

    def __init__(self, dim: int = 512):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        import re
        import zlib

        v = np.zeros(self._dim, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(tok.encode()) % self._dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self._dim), "float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def build_embedder(cfg: dict, test_mode: bool = False):
    if test_mode:
        return HashingEmbedder()
    e = cfg["embedding"]
    return SentenceTransformerEmbedder(e["model_name"], e.get("model_path"),
                                       e.get("query_prefix", ""), e.get("batch_size", 32))
