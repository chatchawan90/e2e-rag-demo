"""Evaluation: retrieval hit-rate, answer correctness, citation quality, and refusals.

Each eval item is small and checkable by code (no LLM judge needed for the core metrics):

  - id: saa-volume-limit
    question: How much hazardous waste can be kept in a satellite accumulation area?
    gold_docs: [epa-saa]            # any of these documents counts as correct retrieval
    must_include: [["55"]]          # every group must match; any alternative in a group is enough
    tags: [en, threshold]
  - id: no-pm25
    question: What is Thailand's 24-hour PM2.5 standard?
    answerable: false               # the corpus does not cover this; the right answer is a refusal

Metrics
  retrieval.hit@k     a gold document appears in the top-k passages given to Claude
  retrieval.mrr       1/rank of the first gold passage
  answer.correct      answered (not refused) and every must_include group matched
  answer.grounded     at least one citation points at a gold document
  answer.cite_prec    share of citations that point at gold documents
  refusal.correct     unanswerable items that were refused (and answerable items that weren't)
"""
from __future__ import annotations

import json
import math
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .answer import answer as run_answer
from .index import Index
from .rewrite import expand_query
from .textnorm import normalize


@dataclass
class Item:
    id: str
    question: str
    gold_docs: list[str] = field(default_factory=list)
    must_include: list[list[str]] = field(default_factory=list)
    answerable: bool = True
    gold_pages: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    expected_scope: str = "in_scope"


def load_items(path: Path) -> list[Item]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = []
    for d in raw["items"]:
        mi = d.get("must_include", [])
        d["must_include"] = [[g] if isinstance(g, str) else list(g) for g in mi]
        if d.get("expected_scope", "in_scope") not in {"in_scope", "out_of_scope", "clarify"}:
            raise ValueError("invalid expected_scope")
        items.append(Item(**d))
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate eval item ids")
    return items


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", normalize(s).lower().replace(",", ""))


def keyword_groups_ok(text: str, groups: list[list[str]]) -> tuple[bool, list[list[str]]]:
    t = _norm(text)
    missing = [g for g in groups if not any(_norm(alt) in t for alt in g)]
    return not missing, missing


# ----------------------------------------------------------------------------- verify

def verify(items: list[Item], index: Index) -> list[dict]:
    """Check the eval set against the corpus before trusting any score.

    For every answerable item: gold docs must be indexed, and each must_include group must
    literally appear in at least one gold document. A failure means the gold label is wrong,
    the document failed to extract, or the keyword needs an alternative spelling.
    """
    problems = []
    doc_text: dict[str, str] = {}
    for c in index.chunks:
        doc_text[c.doc_id] = doc_text.get(c.doc_id, "") + "\n" + c.text
    for it in items:
        if not it.answerable:
            continue
        missing_docs = [d for d in it.gold_docs if d not in doc_text]
        if missing_docs:
            problems.append({"id": it.id, "problem": f"gold docs not indexed: {missing_docs}"})
            continue
        corpus = "\n".join(doc_text[d] for d in it.gold_docs)
        ok, missing = keyword_groups_ok(corpus, it.must_include)
        if not ok:
            problems.append({"id": it.id, "problem": f"keywords not found in gold docs: {missing}"})
    return problems


# ----------------------------------------------------------------------------- run

def document_metrics(retrieved_docs: list[str], gold_docs: list[str], *, k: int) -> dict:
    """Known-gold document coverage and binary nDCG, after the passage cutoff.

    Collapse duplicate documents in their first-seen order among the first k passages.
    These metrics ignore gold_pages; hit/MRR retain their existing page constraints.
    Unjudged documents count as nonrelevant; incomplete labels limit interpretation.
    """
    gold = set(gold_docs)
    if not gold or k <= 0:
        return {}
    ranking = list(dict.fromkeys(retrieved_docs[:k]))
    dcg = sum(1 / math.log2(rank + 1) for rank, did in enumerate(ranking, 1) if did in gold)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return {"doc_recall": len(gold.intersection(ranking)) / len(gold), "doc_ndcg": dcg / ideal}

def run(items: list[Item], index: Index, client, *, model: str, rewrite_model: str, k: int = 8,
        retrieval_only: bool = False, expand: bool = True, log=print, mode: str = "auto",
        lang: str | None = None, doc_ids: set[str] | None = None, jurisdiction: str | None = None,
        metadata: dict[str, str] | None = None, candidate_k: int | None = None,
        per_doc_cap: int = 3, rerank: bool = False, provider: str = "anthropic", answer_fn=None) -> dict:
    mode = index.resolve_mode(mode)
    filters = dict(lang=lang, doc_ids=doc_ids, jurisdiction=jurisdiction, metadata=metadata)
    retrieval = dict(mode=mode, candidate_k=candidate_k, per_doc_cap=per_doc_cap, rerank=rerank)
    rows = []
    for n, it in enumerate(items, 1):
        t0 = time.time()
        row: dict = {"id": it.id, "question": it.question, "tags": it.tags, "answerable": it.answerable,
                     "gold_docs": it.gold_docs, "gold_pages": it.gold_pages, "must_include": it.must_include,
                     "expected_scope": it.expected_scope}
        trace = {}
        if retrieval_only:
            queries = expand_query(it.question, client if expand else None, rewrite_model, provider=provider)
            hits = index.search(queries, k=k, trace=trace, **retrieval, **filters)
            ans = None
        else:
            if provider == "openai" and answer_fn is None:
                from functools import partial
                from .chat import answer as chat_answer
                answer_fn = partial(chat_answer, provider=provider)
            ans = (answer_fn or run_answer)(it.question, index, client, model=model, rewrite_model=rewrite_model,
                                           k=k, expand=expand, trace=trace, **retrieval, **filters)
            hits, queries = ans.hits, ans.queries
        row["queries"] = queries
        row["retrieved"] = [f"{h.chunk.doc_id}:p{h.chunk.page}" for h in hits]
        row["retrieved_chunks"] = [{"chunk_id": h.chunk.id, "doc_id": h.chunk.doc_id,
                                    "page": h.chunk.page, "rank": h.rank, "score": h.score} for h in hits]
        row["trace"] = trace

        if it.answerable and it.gold_docs:
            row.update(document_metrics([h.chunk.doc_id for h in hits], it.gold_docs, k=k))
            ranks = [h.rank for h in hits if h.chunk.doc_id in it.gold_docs
                     and (not it.gold_pages or h.chunk.page in it.gold_pages)]
            row["hit"] = bool(ranks)
            row["rr"] = 1.0 / ranks[0] if ranks else 0.0

        if ans is not None:
            row["scope_decision"] = ans.scope_decision
            row["scope_correct"] = ans.scope_decision == it.expected_scope
            row["answer"] = ans.text
            row["citations"] = [f"{c.doc_id}:p{c.page}" for c in ans.citations]
            row["usage"] = ans.usage
            if it.answerable:
                ok, missing = keyword_groups_ok(ans.text, it.must_include)
                row["refused"] = not ans.found
                row["correct"] = ans.found and ok and row["scope_correct"]
                row["missing_keywords"] = missing
                gold_cites = [c for c in ans.citations if c.doc_id in it.gold_docs]
                row["grounded"] = bool(gold_cites)
                row["cite_prec"] = len(gold_cites) / len(ans.citations) if ans.citations else 0.0
            else:
                row["refused"] = not ans.found and ans.scope_decision != "clarify"
                row["correct"] = not ans.found and row["scope_correct"]
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        status = "" if ans is None else (" ok" if row.get("correct") else " FAIL")
        log(f"[{n}/{len(items)}] {it.id}: hit={row.get('hit', '-')}{status}")
    return {"config": {"model": model, "rewrite_model": rewrite_model, "k": k, "expand": expand,
                       "retrieval_only": retrieval_only, **retrieval, "provider": provider,
                       "scope_routing": not retrieval_only,
                       "document_metrics": "Known gold docs; first k passages, deduplicated by doc; binary relevance; ignores gold_pages",
                       "embedding_model": index.vectors.embedder.model_name if mode != "bm25" else None,
                       "filters": {**filters, "doc_ids": sorted(doc_ids) if doc_ids is not None else None}},
            "summary": summarize(rows), "rows": rows}


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(rows: list[dict]) -> dict:
    def block(rs):
        ans = [r for r in rs if r["answerable"]]
        una = [r for r in rs if not r["answerable"] and r.get("expected_scope", "in_scope") != "clarify"]
        out = {"n": len(rs),
               "retrieval.hit@k": _mean(r["hit"] for r in ans if "hit" in r),
               "retrieval.mrr": _mean(r["rr"] for r in ans if "rr" in r),
               "retrieval.doc_recall@k": _mean(r["doc_recall"] for r in ans if "doc_recall" in r),
               "retrieval.doc_ndcg@k": _mean(r["doc_ndcg"] for r in ans if "doc_ndcg" in r)}
        if any("correct" in r for r in rs):
            out.update({
                "scope.correct": _mean(r["scope_correct"] for r in rs if "scope_correct" in r),
                "answer.correct": _mean(r["correct"] for r in ans),
                "answer.grounded": _mean(r["grounded"] for r in ans),
                "answer.cite_prec": _mean(r["cite_prec"] for r in ans if r["citations"]),
                "answer.false_refusal": _mean(r["refused"] for r in ans),
                "refusal.correct": _mean(r["correct"] for r in una),
            })
        return out
    tags = sorted({t for r in rows for t in r["tags"]})
    return {"overall": block(rows), **{f"tag:{t}": block([r for r in rows if t in r["tags"]]) for t in tags}}


def write_report(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# Eval {stamp}", "", f"Config: `{json.dumps(result['config'])}`", "",
          "| slice | n | hit@k | MRR | doc recall@k | doc nDCG@k | correct | grounded | cite prec | false refusal | refusal ok |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in result["summary"].items():
        md.append("| " + " | ".join([name, str(s["n"])] + [
            "-" if s.get(key) is None else f"{s[key]:.2f}" for key in
            ["retrieval.hit@k", "retrieval.mrr", "retrieval.doc_recall@k", "retrieval.doc_ndcg@k", "answer.correct", "answer.grounded",
             "answer.cite_prec", "answer.false_refusal", "refusal.correct"]]) + " |")
    fails = [r for r in result["rows"] if r.get("correct") is False or r.get("hit") is False]
    if fails:
        md += ["", "## Failures", ""]
        for r in fails:
            md.append(f"- **{r['id']}** hit={r.get('hit')} correct={r.get('correct')} "
                      f"missing={r.get('missing_keywords')} retrieved={r['retrieved'][:4]}")
    path = out_dir / f"eval-{stamp}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- draft

DRAFT_PROMPT = """Below is one passage from a regulatory document ({title}).
Write {n} factual question(s) that this passage alone answers, the kind a compliance officer
would actually ask. For each, give 1-3 short keyword groups that a correct answer must contain
(exact numbers, units, or key terms copied from the passage; list alternative spellings in a group).
Write questions in {lang}. Return JSON only:
[{{"question": "...", "must_include": [["..."], ["...", "alt"]]}}]

Passage (page {page}):
{text}"""


def draft(index: Index, client, *, model: str, doc_ids: list[str] | None, n: int, seed: int = 0) -> list[dict]:
    """Propose eval items from random passages. Output is a DRAFT: a human must review each one."""
    rng = random.Random(seed)
    pool = [c for c in index.chunks if len(c.text) > 300 and (not doc_ids or c.doc_id in doc_ids)]
    out = []
    for c in rng.sample(pool, min(n, len(pool))):
        d = index.docs[c.doc_id]
        msg = client.messages.create(model=model, max_tokens=600, messages=[{"role": "user", "content": DRAFT_PROMPT.format(
            title=d.title, n=1, lang="Thai" if c.lang == "th" else "English", page=c.page, text=c.text)}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        try:
            for q in json.loads(re.search(r"\[.*\]", text, re.S).group()):
                out.append({"id": f"draft-{c.id.replace(':', '-')}", "question": q["question"],
                            "gold_docs": [c.doc_id], "gold_pages": [c.page],
                            "must_include": q.get("must_include", []), "tags": [c.lang, "drafted"],
                            "notes": "DRAFT — review before adding to questions.yaml"})
        except Exception:  # noqa: BLE001
            continue
    return out
