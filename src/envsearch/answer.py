"""Grounded answers with Claude's native citations over search_result content blocks.

Each retrieved chunk becomes one `search_result` block whose `source` is the PDF URL with a
#page anchor. The chunk is split into short text blocks, because Claude cites whole blocks:
smaller blocks give tighter quotes. Claude returns text blocks carrying
`search_result_location` citations, which we turn into numbered footnotes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index import Hit, Index
from .rewrite import expand_query
from .textnorm import is_thai
from .scope import route_question

NOT_FOUND = "I couldn't find this in the indexed documents."
NOT_FOUND_TH = "ไม่พบข้อมูลนี้ในเอกสารที่จัดทำดัชนีไว้"

SYSTEM = f"""You answer questions about environmental regulations using ONLY the search results
provided in the user's message (US EPA guidance and Thai government notifications).

Rules:
- Every factual claim must be supported by the search results. Do not use outside knowledge
  to fill gaps, even if you are confident.
- If the results do not contain the answer, reply with exactly "{NOT_FOUND}"
  (or "{NOT_FOUND_TH}" for a Thai question) and then one sentence on what the documents do cover.
- Answer in the language of the question. When quoting a Thai source in an English answer,
  translate it and keep numbers and units exactly as written.
- Always say which jurisdiction a rule belongs to (US federal vs Thailand). Never apply a US
  rule to Thailand or vice versa.
- Be concise: lead with the direct answer (the number, threshold, or yes/no), then the conditions.
"""

_SENT_EN = re.compile(r"(?<=[.;:?!])\s+(?=[A-Z(§\"'])")


def split_citable(text: str, max_len: int = 350) -> list[str]:
    """Split a chunk into short, self-contained blocks for fine-grained citations."""
    out: list[str] = []
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        pieces = [para] if is_thai(para) else _SENT_EN.split(para)
        for piece in pieces:
            while len(piece) > max_len:
                cut = piece.rfind(" ", 0, max_len)
                cut = cut if cut > max_len // 2 else max_len
                out.append(piece[:cut].strip())
                piece = piece[cut:].strip()
            if piece:
                out.append(piece)
    return out or [text]


def to_search_results(hits: list[Hit], index: Index) -> list[dict]:
    blocks = []
    for h in hits:
        doc = index.docs[h.chunk.doc_id]
        title = f"{doc.title} (p.{h.chunk.page}, {doc.jurisdiction or 'n/a'})"
        blocks.append({
            "type": "search_result",
            "source": index.cite_url(h.chunk),
            "title": title,
            "content": [{"type": "text", "text": t} for t in split_citable(h.chunk.text)],
            "citations": {"enabled": True},
        })
    return blocks


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc_id: str
    page: int
    title: str
    url: str
    cited_text: str


@dataclass
class Answer:
    question: str
    text: str                                 # answer with [n] footnote markers
    citations: list[Citation]
    hits: list[Hit]
    queries: list[str]
    found: bool
    usage: dict = field(default_factory=dict)
    scope_decision: str = "in_scope"

    def render(self) -> str:
        lines = [self.text.strip(), ""]
        if self.citations:
            lines.append("Sources:")
            for c in self.citations:
                quote = c.cited_text.strip().replace("\n", " ")
                quote = quote if len(quote) <= 220 else quote[:217] + "..."
                lines.append(f"  [{c.n}] {c.title} — {c.url}\n       \"{quote}\"")
        return "\n".join(lines)


def parse_response(content, hits: list[Hit], index: Index) -> tuple[str, list[Citation]]:
    """Turn Claude's text blocks + search_result_location citations into footnoted text."""
    parts: list[str] = []
    cites: list[Citation] = []
    key_to_n: dict[tuple, int] = {}
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        parts.append(block.text)
        markers = []
        for c in getattr(block, "citations", None) or []:
            if getattr(c, "type", None) != "search_result_location":
                continue
            i = c.search_result_index
            if not (0 <= i < len(hits)):
                continue  # defensive: should never happen, we sent exactly len(hits) results
            key = (i, c.start_block_index, c.end_block_index)
            if key not in key_to_n:
                ch = hits[i].chunk
                key_to_n[key] = len(cites) + 1
                cites.append(Citation(
                    n=key_to_n[key], chunk_id=ch.id, doc_id=ch.doc_id, page=ch.page,
                    title=index.docs[ch.doc_id].title, url=index.cite_url(ch), cited_text=c.cited_text,
                ))
            markers.append(key_to_n[key])
        if markers:
            parts.append("".join(f"[{n}]" for n in dict.fromkeys(markers)))
    return "".join(parts), cites


def answer(question: str, index: Index, client, *, model: str, rewrite_model: str | None = None,
           k: int = 8, expand: bool = True, lang: str | None = None, mode: str = "auto",
           doc_ids: set[str] | None = None, jurisdiction: str | None = None,
           metadata: dict[str, str] | None = None, candidate_k: int | None = None,
           per_doc_cap: int = 3, rerank: bool = False, trace: dict | None = None) -> Answer:
    route = route_question(question, client, model=rewrite_model or model, trace=trace)
    if route.decision != "in_scope":
        return Answer(question, route.reply(question), [], [], [], False, scope_decision=route.decision)
    index.resolve_mode(mode)
    queries = expand_query(route.query, client if expand else None, rewrite_model or model)
    hits = index.search(queries, k=k, lang=lang, mode=mode, doc_ids=doc_ids,
                        jurisdiction=jurisdiction, metadata=metadata, candidate_k=candidate_k,
                        per_doc_cap=per_doc_cap, rerank=rerank, trace=trace)
    if not hits:
        msg = NOT_FOUND_TH if is_thai(question) else NOT_FOUND
        return Answer(question, msg, [], [], queries, found=False)

    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM,
        messages=[{"role": "user", "content": [
            *to_search_results(hits, index),
            {"type": "text", "text": question},
        ]}],
    )
    text, cites = parse_response(resp.content, hits, index)
    refused = text.strip().startswith((NOT_FOUND, NOT_FOUND_TH))
    usage = {}
    if getattr(resp, "usage", None) is not None:
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return Answer(question, text, cites, hits, queries, found=bool(cites) and not refused, usage=usage)
