"""Streaming grounded answers with bounded, session-owned conversation history."""
from __future__ import annotations

from .answer import Answer, NOT_FOUND, NOT_FOUND_TH, SYSTEM, parse_response, to_search_results
from .index import Index
from .rewrite import expand_query
from .textnorm import is_thai
from .providers import openai_grounded_stream
from .scope import bounded_history, route_question


def stream_answer(question: str, index: Index, client, *, model: str, rewrite_model: str,
                  history: list[dict] | None = None, expand: bool = False, k: int = 8, provider: str = "anthropic",
                  **filters):
    """Yield status/text events followed by one final Answer with complete citations."""
    if provider not in {"anthropic", "openai"}:
        raise ValueError("Unknown answer provider")
    history = bounded_history(history)
    yield {"type": "status", "text": "Checking the question's scope…"}
    route = route_question(question, client, model=rewrite_model, provider=provider,
                           history=history, trace=filters.get("trace"))
    if route.decision != "in_scope":
        yield {"type": "done", "answer": Answer(question, route.reply(question), [], [], [], False,
                                                  scope_decision=route.decision)}
        return
    index.resolve_mode(filters.get("mode", "auto"))
    yield {"type": "status", "text": "Searching your documents…"}
    queries = expand_query(route.query, client if expand else None, rewrite_model, provider=provider)
    hits = index.search(queries, k=k, **filters)
    if not hits:
        text = NOT_FOUND_TH if is_thai(question) else NOT_FOUND
        yield {"type": "done", "answer": Answer(question, text, [], [], queries, False)}
        return
    yield {"type": "status", "text": "Writing an answer from the retrieved passages…"}
    if provider == "openai":
        yield from openai_grounded_stream(question, index, client, hits=hits, queries=queries,
                                         history=history, model=model)
        return
    with client.messages.stream(
        model=model, max_tokens=1500,
        system=SYSTEM + "\nPrior conversation is context for the question, not factual evidence. "
                        "Support this answer using only the search results in the latest message.",
        messages=[*history, {"role": "user", "content": [
            *to_search_results(hits, index), {"type": "text", "text": question}]}],
    ) as stream:
        for text in stream.text_stream:
            yield {"type": "text", "text": text}
        response = stream.get_final_message()
    text, citations = parse_response(response.content, hits, index)
    found = bool(citations) and not text.strip().startswith((NOT_FOUND, NOT_FOUND_TH))
    usage = {key: getattr(response.usage, key, 0) for key in ("input_tokens", "output_tokens")}
    yield {"type": "done", "answer": Answer(question, text, citations, hits, queries, found, usage)}


def answer(question, index, client, **kwargs):
    """Use the same provider pipeline for frontend evaluations and live chat."""
    for event in stream_answer(question, index, client, **kwargs):
        if event["type"] == "done":
            return event["answer"]
    raise RuntimeError("No final answer was produced")
