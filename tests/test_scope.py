import json
from types import SimpleNamespace as NS

import pytest

from envsearch.chat import stream_answer
from envsearch.answer import answer
from tests.conftest import response, text_block


def client_for(decision, query=""):
    return NS(messages=NS(create=lambda **_: response(text_block(json.dumps({"decision": decision, "query": query})))))


@pytest.mark.parametrize("question,decision", [("ผมชื่ออะไร", "out_of_scope"),
                                               ("Tell me a joke", "out_of_scope"),
                                               ("What about Thailand?", "clarify")])
@pytest.mark.parametrize("streaming", [True, False])
def test_blocked_routes_never_retrieve_expand_or_answer(monkeypatch, question, decision, streaming):
    class NoRetrieval:
        def __getattr__(self, name):
            pytest.fail(f"Index accessed before scope check: {name}")
    trace = {}
    kw = dict(model="m", rewrite_model="r", expand=True, trace=trace)
    if streaming:
        ans = list(stream_answer(question, NoRetrieval(), client_for(decision), **kw))[-1]["answer"]
    else:
        ans = answer(question, NoRetrieval(), client_for(decision), **kw)
    assert ans.scope_decision == decision
    assert ans.text and not ans.citations and not ans.hits and not ans.queries
    assert trace["scope_decision"] == decision and trace["returned"] == 0
    if question == "ผมชื่ออะไร":
        assert "นอกขอบเขต" in ans.text


def test_followup_scope_uses_recent_history_and_standalone_query(index):
    from envsearch.scope import route_question
    history = [{"role": "user", "content": "What is the factory effluent BOD limit?"}]
    def create(**kwargs):
        payload = json.loads(kwargs["messages"][0]["content"])
        assert payload["history"] == history
        assert payload["question"] == "What about Thailand?"
        return response(text_block('{"decision":"in_scope","query":"factory effluent BOD limit in Thailand"}'))
    route = route_question("What about Thailand?", NS(messages=NS(create=create)), model="r", history=history)
    assert route.decision == "in_scope" and route.query.endswith("Thailand")


@pytest.mark.parametrize("payload", ['{}', '{"decision":"maybe"}', 'plain prose',
                                     '{"decision":"in_scope","query":""}'])
def test_invalid_router_output_fails_without_search(payload):
    from envsearch.scope import ScopeRoutingError
    client = NS(messages=NS(create=lambda **_: response(text_block(payload))))
    with pytest.raises(ScopeRoutingError):
        list(stream_answer("What is the limit?", None, client, model="m", rewrite_model="r"))


def test_router_failure_does_not_fall_through_to_search():
    def fail(**_):
        raise RuntimeError("router unavailable")
    with pytest.raises(RuntimeError):
        list(stream_answer("What is the limit?", None, NS(messages=NS(create=fail)), model="m", rewrite_model="r"))


def test_in_scope_missing_evidence_is_not_outside_scope(index):
    trace = {}
    ans = list(stream_answer("PM2.5 standard", index, client_for("in_scope", "PM2.5 standard"),
                             model="m", rewrite_model="r", doc_ids=set(), trace=trace))[-1]["answer"]
    assert ans.scope_decision == "in_scope" and not ans.found
    assert "couldn't find" in ans.text and trace["scope_decision"] == "in_scope"


def test_eval_does_not_reward_wrong_scope_rejection(index):
    from envsearch.evaluate import Item, run
    common = dict(model="m", rewrite_model="r", expand=False, log=lambda *_: None)
    result = run([Item("missing", "PM2.5 standard", answerable=False)], index,
                 client_for("out_of_scope"), **common)
    assert result["summary"]["overall"]["scope.correct"] == 0
    assert result["rows"][0]["correct"] is False
    result = run([Item("name", "ผมชื่ออะไร", answerable=False, expected_scope="out_of_scope")], index,
                 client_for("out_of_scope"), **common)
    assert result["rows"][0]["correct"] is True and result["rows"][0]["retrieved"] == []


def test_scope_trace_does_not_persist_user_text(tmp_path):
    from envsearch.scope import route_question
    from envsearch.monitoring import record_event, recent_events
    trace = {}
    route_question("private name", client_for("out_of_scope"), model="r", trace=trace)
    record_event(tmp_path, "chat", "ok", .1, trace=trace)
    row = recent_events(tmp_path)[0]
    assert row["trace"]["scope_decision"] == "out_of_scope"
    assert row["trace"]["scope_input_tokens"] == 100
    assert "private name" not in str(row)


def test_correct_clarification_does_not_inflate_refusal_score(index):
    from envsearch.evaluate import Item, run
    result = run([Item("unclear", "What about Thailand?", answerable=False, expected_scope="clarify")],
                 index, client_for("clarify"), model="m", rewrite_model="r", expand=False, log=lambda *_: None)
    assert result["rows"][0]["refused"] is False
    assert result["summary"]["overall"]["refusal.correct"] is None
    assert result["summary"]["overall"]["scope.correct"] == 1
