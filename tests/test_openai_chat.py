from types import SimpleNamespace as NS

import pytest

from envsearch.chat import stream_answer
from tests.conftest import scope_json


def responding_with(stream):
    def create(**kw):
        if not kw.get("stream"):
            return NS(status="completed", output_text=scope_json(kw))
        return stream
    return create


class Stream:
    def __init__(self, text="Up to 55 gallons.[1]", status="completed"):
        self.text, self.status = text, status
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def __iter__(self):
        yield NS(type="response.output_text.delta", delta=self.text)
        yield NS(type=f"response.{self.status}", response=NS(
            status=self.status, output_text=self.text, usage=NS(input_tokens=120, output_tokens=15)))


def test_openai_stream_citations_and_privacy(index):
    def create(**kw):
        if not kw.get("stream"):
            return NS(status="completed", output_text=scope_json(kw))
        assert kw["store"] is False and kw["stream"] is True
        assert "fx-saa" in str(kw["input"])
        return Stream()
    events = list(stream_answer("satellite accumulation gallons", index, NS(responses=NS(create=create)),
                               provider="openai", model="gpt-4.1-mini", rewrite_model="r"))
    ans = events[-1]["answer"]
    assert ans.found and ans.text.endswith("[1]")
    assert ans.citations[0].doc_id == "fx-saa"
    assert ans.usage == {"input_tokens": 120, "output_tokens": 15}


def test_openai_unknown_citation_is_not_accepted(index):
    client = NS(responses=NS(create=responding_with(Stream("A claim.[999]"))))
    ans = list(stream_answer("satellite accumulation", index, client, provider="openai",
                             model="m", rewrite_model="r"))[-1]["answer"]
    assert not ans.found and not ans.citations
    assert "[999]" not in ans.text


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_openai_incomplete_response_not_saved_as_success(index, status):
    client = NS(responses=NS(create=responding_with(Stream(status=status))))
    with pytest.raises(RuntimeError):
        list(stream_answer("satellite accumulation", index, client, provider="openai", model="m", rewrite_model="r"))


def test_openai_followup_and_expansion_use_selected_provider(index):
    calls = []
    def create(**kw):
        calls.append(kw)
        if kw.get("stream"):
            return Stream()
        if len(calls) == 1:
            return NS(status="completed", output_text='{"decision":"in_scope","query":"satellite accumulation gallons"}')
        return NS(status="completed", output_text='{"en":["satellite accumulation"],"th":[]}')
    client = NS(responses=NS(create=create))
    ans = list(stream_answer("How much?", index, client, provider="openai", model="m", rewrite_model="r",
                             expand=True, history=[{"role":"user", "content":"satellite accumulation"}]))[-1]["answer"]
    assert len(calls) == 3 and ans.found
    assert all(c["store"] is False for c in calls)
    assert ans.queries[0] == "satellite accumulation gallons"


def test_real_sdk_parses_stream_without_network(index):
    import json
    import httpx
    openai = pytest.importorskip("openai")
    def respond(request):
        body = json.loads(request.content)
        if not body.get("stream"):
            return httpx.Response(200, json={"id":"resp_route", "object":"response", "created_at":1,
                "status":"completed", "model":"gpt-4.1-mini", "output":[{"type":"message", "id":"msg_route",
                "role":"assistant", "status":"completed", "content":[{"type":"output_text",
                "text":scope_json(body), "annotations":[]}]}]})
        assert body["store"] is False and body["stream"] is True
        response = {"id": "resp_fixture", "object": "response", "created_at": 1,
                    "status": "completed", "model": "gpt-4.1-mini",
                    "output": [{"type": "message", "id": "msg_fixture", "role": "assistant", "status": "completed",
                                "content": [{"type": "output_text", "text": "55 gallons.[1]", "annotations": []}]}],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        events = [{"type": "response.output_text.delta", "delta": "55 gallons.[1]"},
                  {"type": "response.completed", "response": response}]
        data = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=data)
    with openai.OpenAI(api_key="test-not-real", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        answer = list(stream_answer("satellite accumulation gallons", index, client, provider="openai",
                                     model="gpt-4.1-mini", rewrite_model="r"))[-1]["answer"]
    assert answer.found and answer.usage["input_tokens"] == 10


def test_provider_refusal_is_visible_not_blank(index):
    class RefusalStream(Stream):
        def __iter__(self):
            yield NS(type="response.refusal.delta", delta="I cannot assist with that request.")
            yield NS(type="response.completed", response=NS(status="completed", output_text="",
                output=[NS(content=[NS(type="refusal", refusal="I cannot assist with that request.")])],
                usage=NS(input_tokens=10, output_tokens=5)))
    client = NS(responses=NS(create=responding_with(RefusalStream())))
    events = list(stream_answer("satellite accumulation", index, client, provider="openai", model="m", rewrite_model="r"))
    assert any(e.get("text") == "I cannot assist with that request." for e in events)
    assert events[-1]["answer"].text == "I cannot assist with that request."
    assert not events[-1]["answer"].found
