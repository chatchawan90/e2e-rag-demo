"""Bilingual query expansion with a small Claude model.

An English question about Thai effluent limits should find Thai text, and a Thai question
about EPA rules should find English text. We ask a cheap model for 1-2 keyword-style
queries in each language. If the API is unavailable the original query is used alone.
"""
from __future__ import annotations

import json
import re
from .providers import text_completion

PROMPT = """You rewrite a user's question into search queries for a keyword (BM25) index of
environmental regulations. The index holds English US EPA hazardous-waste guidance and Thai
Ministry of Industry / MNRE notifications (ประกาศ) on industrial waste and effluent.

Return JSON only: {{"en": [..], "th": [..]}} with 1-2 short keyword queries per language.
Use the regulator's own vocabulary (e.g. "satellite accumulation area", "large quantity generator",
"มาตรฐานควบคุมการระบายน้ำทิ้ง", "ผู้ก่อกำเนิดสิ่งปฏิกูล", "บีโอดี"). Do not answer the question.

Question: {question}"""


def expand_query(question: str, client, model: str, *, provider: str = "anthropic") -> list[str]:
    queries = [question]
    if client is None:
        return queries
    try:
        text = text_completion(
            client, provider=provider, model=model, max_tokens=300,
            messages=[{"role": "user", "content": PROMPT.format(question=question)}],
        )
        data = json.loads(re.search(r"\{.*\}", text, re.S).group())
        for lang in ("en", "th"):
            queries.extend(q for q in data.get(lang, [])[:2] if isinstance(q, str))
    except Exception:  # noqa: BLE001 — expansion is best-effort
        pass
    return list(dict.fromkeys(queries))
