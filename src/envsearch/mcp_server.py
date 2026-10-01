"""MCP server so Claude Desktop (or any MCP client) can search the corpus.

In Claude Desktop, Claude itself is the answering model, so the main tool returns passages
with document, page and URL labels and Claude cites them in its reply. `ask_with_citations`
is the alternative path: it runs this repo's own pipeline (API-native citations) and returns
a finished, footnoted answer — useful for comparing the two.

Runs over stdio. Logs go to stderr only: stdout is the protocol channel.
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import Literal

from mcp.server.fastmcp import FastMCP

from . import OWNER_NOTICE
from .answer import answer as run_answer
from .config import get_client, get_settings
from .index import Index

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("envsearch.mcp")

mcp = FastMCP("envsearch", instructions=OWNER_NOTICE + ". Search environmental regulations and cite the original sources.")


@lru_cache(maxsize=2)
def _index(load_vectors: bool = True) -> Index:
    s = get_settings()
    idx = Index.load(s.index_dir, load_vectors=load_vectors)
    log.info("loaded %d chunks from %s", len(idx.chunks), s.index_dir)
    return idx


def _format_hits(hits, idx: Index) -> str:
    if not hits:
        return "No matching passages. Try regulator vocabulary, or search again in the other language (English/Thai)."
    out = []
    for h in hits:
        d = idx.docs[h.chunk.doc_id]
        out.append(
            f"[{h.rank}] {d.title}\n"
            f"    doc_id={d.id} page={h.chunk.page} jurisdiction={d.jurisdiction} lang={h.chunk.lang} chunk_id={h.chunk.id}\n"
            f"    url={idx.cite_url(h.chunk)}\n"
            f"{h.chunk.text}\n"
        )
    return "\n".join(out)


@mcp.tool()
def search_regulations(query: str, k: int = 6, language: Literal["any", "en", "th"] = "any",
                       doc_ids: list[str] | None = None, jurisdiction: str | None = None,
                       metadata: dict[str, str] | None = None,
                       mode: Literal["auto", "bm25", "vector", "hybrid"] = "auto",
                       rerank: bool = False, candidate_k: int = 32, per_doc_cap: int = 3) -> str:
    """Search US EPA hazardous-waste guidance (English) and Thai industrial waste /
    effluent notifications (Thai). Returns passages labelled with document title, page and URL.

    auto uses multilingual hybrid search when vectors were built, otherwise keywords (BM25).
    With BM25, also search in the source language (e.g. "มาตรฐานควบคุมการระบายน้ำทิ้ง บีโอดี").
    jurisdiction filters the regulator's country (US/TH), independently of passage language.
    metadata is an AND of exact matches on custom document fields. doc_ids restricts allowed documents.
    Keep language=any for cross-language retrieval. Cite passages as (title, p.N) with their URL;
    say plainly when the passages don't contain the answer.
    """
    idx = _index(load_vectors=mode != "bm25")
    hits = idx.search([query], k=max(1, min(k, 15)), lang=None if language == "any" else language,
                      doc_ids=set(doc_ids) if doc_ids is not None else None,
                      jurisdiction=jurisdiction, metadata=metadata, mode=mode, rerank=rerank,
                      candidate_k=max(15, min(candidate_k, 200)), per_doc_cap=max(1, per_doc_cap))
    return _format_hits(hits, idx)


@mcp.tool()
def get_passage(chunk_id: str, context: int = 1) -> str:
    """Return a passage by chunk_id plus `context` neighbouring passages from the same document,
    for when a search hit is cut off mid-table or mid-clause."""
    idx = _index(load_vectors=False)
    chunks = idx.neighbors(chunk_id, span=max(0, min(context, 3)))
    if not chunks:
        return f"Unknown chunk_id {chunk_id!r}."
    d = idx.docs[chunks[0].doc_id]
    body = "\n---\n".join(f"(p.{c.page}, {c.id})\n{c.text}" for c in chunks)
    return f"{d.title}\n{d.url}\n\n{body}"


@mcp.tool()
def list_documents() -> str:
    """List the indexed documents with ids, jurisdiction, language and source URL."""
    idx = _index(load_vectors=False)
    counts: dict[str, int] = {}
    for c in idx.chunks:
        counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
    rows = [f"- {d.id} [{d.jurisdiction}/{d.lang}] {d.title}"
            + (f" ({d.title_en})" if d.title_en else "")
            + f" — {counts.get(d.id, 0)} passages — {d.url}"
            for d in idx.docs.values()]
    return "\n".join(rows)


@mcp.tool()
def ask_with_citations(question: str, language: Literal["any", "en", "th"] = "any",
                       doc_ids: list[str] | None = None, jurisdiction: str | None = None,
                       metadata: dict[str, str] | None = None,
                       mode: Literal["auto", "bm25", "vector", "hybrid"] = "auto",
                       expand: bool = True) -> str:
    """Answer a question with this server's own grounded pipeline (bilingual retrieval + Claude API
    citations). Returns a finished answer with numbered sources. Needs ANTHROPIC_API_KEY set in the
    server's environment. Filters and mode match search_regulations; set expand=False to
    skip bilingual query rewriting and measure hybrid retrieval on the original question."""
    client = get_client()
    if client is None:
        return "ask_with_citations needs ANTHROPIC_API_KEY in the MCP server env. Use search_regulations instead."
    s = get_settings()
    a = run_answer(question, _index(load_vectors=mode != "bm25"), client,
                   model=s.answer_model, rewrite_model=s.rewrite_model, k=s.top_k,
                   lang=None if language == "any" else language,
                   doc_ids=set(doc_ids) if doc_ids is not None else None,
                   jurisdiction=jurisdiction, metadata=metadata, mode=mode, expand=expand)
    return a.render()


def main() -> None:
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
