from types import SimpleNamespace as NS

from tests.conftest import response, text_block, sr_citation, scope_response


class Stream:
    text_stream = iter(["Up to ", "55 gallons."])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def get_final_message(self):
        return response(text_block("Up to 55 gallons.", [sr_citation(0)]))


def test_stream_chat_yields_text_then_page_citations(index):
    from envsearch.chat import stream_answer
    client = NS(messages=NS(stream=lambda **_: Stream(), create=lambda **kw: scope_response(kw)))
    events = list(stream_answer("satellite accumulation gallons", index, client, model="m", rewrite_model="r"))
    assert [e["text"] for e in events if e["type"] == "text"] == ["Up to ", "55 gallons."]
    answer = events[-1]["answer"]
    assert answer.citations[0].doc_id == "fx-saa"
    assert answer.citations[0].url.endswith("#page=1") and "[1]" in answer.text


def test_followup_question_uses_history_for_retrieval(index):
    from envsearch.chat import stream_answer
    history = [{"role": "user", "content": "Tell me about satellite accumulation areas."},
               {"role": "assistant", "content": "They hold hazardous waste near its generation point."}]

    def create(**kw):
        assert "satellite accumulation" in str(kw["messages"])
        return response(text_block('{"decision":"in_scope","query":"satellite accumulation area gallon limit"}'))

    def stream(**kw):
        assert kw["messages"][:2] == history
        assert kw["messages"][-1]["content"][-1]["text"] == "How much can they hold?"
        return Stream()

    events = list(stream_answer("How much can they hold?", index, NS(messages=NS(create=create, stream=stream)),
                               model="m", rewrite_model="r", history=history))
    assert events[-1]["answer"].hits[0].chunk.doc_id == "fx-saa"


def test_stream_chat_no_hits_does_not_call_model(index):
    from envsearch.chat import stream_answer
    client = NS(messages=NS(create=lambda **kw: scope_response(kw)))
    events = list(stream_answer("anything", index, client, model="m", rewrite_model="r", doc_ids=set()))
    assert events[-1]["type"] == "done" and not events[-1]["answer"].found
