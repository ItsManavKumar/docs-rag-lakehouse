"""Evaluation: retrieval hit rate@k and answer accuracy.

questions.jsonl, one object per line:
  id, question, answerable (bool), expected_answer, expected_file,
  expected_pages (list[int], optional), expected_keywords (list[str], optional)
"""
from __future__ import annotations

import json
import re

from .rag import NOT_FOUND, answer_question

JUDGE_PROMPT = """You GRADE answers from a document question-answering system.
Given a question, a reference answer written by a human from the source document, and the system's answer,
decide whether the system's answer is correct: it must state the key fact(s) of the reference answer
(numbers and units must match; extra correct detail is fine) and must not contradict it.
Reply with JSON only: {"correct": true|false, "reason": "<one short sentence>"}"""


# Spark schema for eval rows (explicit, because booleans can be None for unanswerable questions)
EVAL_DDL = ("run STRING, id STRING, question STRING, answerable BOOLEAN, k INT, metadata_filter BOOLEAN, "
            "retrieval_hit BOOLEAN, answer_correct BOOLEAN, judge_reason STRING, keywords_found BOOLEAN, "
            "cites_expected_file BOOLEAN, answer STRING, filters_applied STRING, top_files STRING")


def load_questions(path: str) -> list[dict]:
    with open(path) as f:
        qs = [json.loads(ln) for ln in f if ln.strip() and not ln.lstrip().startswith("//")]
    for q in qs:
        q.setdefault("answerable", True)
        q.setdefault("expected_pages", [])
        q.setdefault("expected_keywords", [])
    return qs


def retrieval_hit(q: dict, retrieved: list[dict], k: int) -> bool | None:
    if not q["answerable"]:
        return None
    for r in retrieved[:k]:
        if r["file"] != q["expected_file"]:
            continue
        if not q["expected_pages"]:
            return True
        if any(r["page_start"] <= p <= r["page_end"] for p in q["expected_pages"]):
            return True
    return False


def judge(q: dict, answer: str, llm, judge_model: str | None = None) -> tuple[bool, str]:
    said_not_found = NOT_FOUND in answer.lower()
    if not q["answerable"]:
        return said_not_found, "unanswerable: correct only if it says not in the documents"
    if said_not_found:
        return False, "answerable question, but system said not in the documents"
    msgs = [{"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Question: {q['question']}\nReference answer: {q['expected_answer']}\n"
                                        f"System answer: {answer}"}]
    raw = llm.chat(msgs, model=judge_model)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        verdict = json.loads(m.group(0)) if m else {}
        return bool(verdict.get("correct")), str(verdict.get("reason", ""))
    except json.JSONDecodeError:
        return False, f"unparseable judge output: {raw[:100]}"


def run_eval(questions: list[dict], retriever, llm, k: int = 5, use_filter: bool = True,
             judge_model: str | None = None, run_name: str = "") -> list[dict]:
    rows = []
    for q in questions:
        res = answer_question(q["question"], retriever, llm, k=k, use_filter=use_filter)
        ok, reason = judge(q, res["answer"], llm, judge_model)
        kw = q["expected_keywords"]
        kw_ok = all(w.lower() in res["answer"].lower() for w in kw) if kw else None
        cited_files = {c["file"] for c in res["citations"]}
        rows.append({
            "run": run_name, "id": q["id"], "question": q["question"], "answerable": q["answerable"],
            "k": k, "metadata_filter": use_filter,
            "retrieval_hit": retrieval_hit(q, res["retrieved"], k),
            "answer_correct": ok, "judge_reason": reason, "keywords_found": kw_ok,
            "cites_expected_file": (q.get("expected_file") in cited_files) if q["answerable"] else None,
            "answer": res["answer"], "filters_applied": json.dumps(res["filters_applied"]),
            "top_files": json.dumps([r["file"] for r in res["retrieved"]]),
        })
    return rows


def _rate(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    return {
        "run": rows[0]["run"] if rows else "",
        "k": rows[0]["k"] if rows else None,
        "metadata_filter": rows[0]["metadata_filter"] if rows else None,
        "n_questions": len(rows),
        "retrieval_hit_rate": _rate(r["retrieval_hit"] for r in ans),
        "answer_accuracy": _rate(r["answer_correct"] for r in rows),
        "answerable_accuracy": _rate(r["answer_correct"] for r in ans),
        "unanswerable_refusal_rate": _rate(r["answer_correct"] for r in unans),
        "citation_accuracy": _rate(r["cites_expected_file"] for r in ans),
    }


def markdown_table(summaries: list[dict]) -> str:
    cols = ["run", "k", "metadata_filter", "n_questions", "retrieval_hit_rate", "answer_accuracy",
            "unanswerable_refusal_rate", "citation_accuracy"]
    fmt = lambda v: f"{v:.0%}" if isinstance(v, float) else ("-" if v is None else str(v))
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(s[c]) for c in cols) + " |" for s in summaries]
    return "\n".join(out)
