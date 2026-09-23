"""Retrieval + grounded answering with citations."""
from __future__ import annotations

import re

from .filters import detect_filters, strip_entities

NOT_FOUND = "not in the documents"

SYSTEM_PROMPT = f"""You answer questions about product documents using ONLY the numbered sources provided.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge, even if you are confident.
- After every sentence that states a fact, cite its source tag(s), e.g. [S2] or [S1][S3].
- If the sources do not contain the answer, reply exactly: "That information is {NOT_FOUND}." You may add one short sentence saying what the sources do cover.
- If sources disagree or describe different products, say so and name the product each fact belongs to.
- Keep units and numbers exactly as written in the source. Be concise: 1-4 sentences."""


def format_sources(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        pages = (f"p.{c['page_start']}" if c["page_start"] == c["page_end"]
                 else f"pp.{c['page_start']}-{c['page_end']}")
        parts.append(f"[S{i}] {c['brand']} {c['product']} ({c['doc_type']}), "
                     f"file {c['file']}, {pages}, section: {c['section']}\n{c['text']}")
    return "\n\n".join(parts)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    if chunks:
        user = f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"
    else:
        user = f"Sources: (none retrieved)\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cited_sources(answer: str, chunks: list[dict]) -> list[dict]:
    ids = sorted({int(n) for n in re.findall(r"\[S(\d+)\]", answer)})
    return [{"tag": f"S{i}", "file": chunks[i - 1]["file"], "page_start": chunks[i - 1]["page_start"],
             "page_end": chunks[i - 1]["page_end"], "product": chunks[i - 1]["product"],
             "section": chunks[i - 1]["section"], "source_url": chunks[i - 1].get("source_url", "")}
            for i in ids if 1 <= i <= len(chunks)]


class Retriever:
    def __init__(self, index, embedder, top_k: int = 5, use_metadata_filter: bool = True,
                 hybrid: bool = True):
        self.index = index
        self.embedder = embedder
        self.top_k = top_k
        self.use_metadata_filter = use_metadata_filter
        # hybrid = vector + BM25 keyword search merged (see index.search_hybrid)
        self.hybrid = hybrid and hasattr(index, "search_hybrid")
        self.products = sorted({r["product"] for r in index.records if r.get("product")})
        self.brands = sorted({r["brand"] for r in index.records if r.get("brand") and r["brand"] != "Unknown"})

    def retrieve(self, question: str, k: int | None = None, use_filter: bool | None = None,
                 hybrid: bool | None = None) -> tuple[list[dict], dict]:
        k = k or self.top_k
        use_filter = self.use_metadata_filter if use_filter is None else use_filter
        hybrid = self.hybrid if hybrid is None else (hybrid and hasattr(self.index, "search_hybrid"))
        qv = self.embedder.embed_query(question)
        filters = detect_filters(question, self.products, self.brands) if use_filter else {}
        applied: dict = {}
        if filters.get("product"):
            applied = {"product": filters["product"]}
        elif filters.get("brand"):
            applied = {"brand": filters["brand"]}
        lexical_q = strip_entities(question, filters.get("product", []) + filters.get("brand", []))

        def run(filters):
            if hybrid:
                return self.index.search_hybrid(qv, lexical_q, k, filters)
            return self.index.search(qv, k, filters)

        hits = run(applied or None)
        if applied and not hits:  # filter matched nothing -> fall back to unfiltered
            applied = {}
            hits = run(None)
        return hits, applied


def answer_question(question: str, retriever: Retriever, llm, k: int | None = None,
                    use_filter: bool | None = None, hybrid: bool | None = None) -> dict:
    chunks, applied = retriever.retrieve(question, k, use_filter, hybrid)
    answer = llm.chat(build_messages(question, chunks)).strip()
    return {
        "question": question,
        "answer": answer,
        "not_found": NOT_FOUND in answer.lower(),
        "citations": cited_sources(answer, chunks),
        "filters_applied": applied,
        "retrieved": [{k2: c.get(k2) for k2 in ("rank", "score", "found_by", "file", "product",
                                                 "section", "page_start", "page_end", "chunk_id")}
                      for c in chunks],
    }


def pretty(result: dict) -> str:
    lines = [f"Q: {result['question']}", f"A: {result['answer']}"]
    if result["filters_applied"]:
        lines.append(f"   (metadata filter: {result['filters_applied']})")
    for c in result["citations"]:
        pages = (f"p.{c['page_start']}" if c["page_start"] == c["page_end"]
                 else f"pp.{c['page_start']}-{c['page_end']}")
        lines.append(f"   [{c['tag']}] {c['file']} {pages} - {c['section']}")
    return "\n".join(lines)
