from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src/envsearch/web_app.py"


def enable_scoped_passage_search(app, monkeypatch):
    import anthropic
    from contextlib import nullcontext
    from tests.conftest import fake_client
    def no_answer(_):
        pytest.fail("Passage search should only call the scope classifier")
    monkeypatch.setattr(anthropic, "Anthropic", lambda **_: nullcontext(fake_client(no_answer)))
    next(t for t in app.text_input if t.label == "Anthropic API key").set_value("test-scope-key").run()
    app.radio[0].set_value("Find passages").run()


def test_ui_requires_scope_key_in_both_modes(index, tmp_path, monkeypatch):
    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=10)
    assert not app.exception
    assert app.radio[0].value == "Chat with sources"
    assert app.chat_input[0].disabled
    app.radio[0].set_value("Find passages").run()
    assert app.chat_input[0].disabled
    assert any("scope check" in m.value for m in app.info)


def test_ui_empty_library_offers_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=10)
    assert not app.exception
    assert app.chat_input[0].disabled
    assert any("Add a PDF" in message.value for message in app.info)


def test_ui_streams_answer_and_keeps_citations(index, tmp_path, monkeypatch):
    import anthropic
    from types import SimpleNamespace as NS
    from tests.conftest import response, text_block, sr_citation, scope_response

    class Stream:
        text_stream = ["Up to ", "55 gallons."]
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def get_final_message(self):
            return response(text_block("Up to 55 gallons.", [sr_citation(0)]))

    class Client:
        messages = NS(stream=lambda **_: Stream(), create=lambda **kw: scope_response(kw))
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass

    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(anthropic, "Anthropic", lambda **_: Client())
    app = AppTest.from_file(str(APP)).run(timeout=10)
    app.chat_input[0].set_value("satellite accumulation gallons").run(timeout=10)
    assert not app.exception
    assert any("55 gallons.[1]" in m.value for m in app.markdown)
    assert len(app.session_state["messages"]) == 2


def test_ui_evaluation_gold_and_monitoring(index, tmp_path, monkeypatch):
    import json
    from envsearch import evaluate
    from envsearch.monitoring import recent_events
    item = evaluate.Item("fixture-question", "satellite accumulation gallons",
                         gold_docs=["fx-saa"], must_include=[["55"]], tags=["en"])
    monkeypatch.setattr(evaluate, "load_items", lambda _: [item])
    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=10)
    assert not app.exception
    assert {t.label for t in app.tabs} >= {"Evaluation", "Gold dataset", "Monitoring", "Tools"}
    next(b for b in app.button if b.label == "Run test").click().run(timeout=10)
    assert not app.exception
    paths = list((tmp_path / "evals").glob("*.json"))
    assert len(paths) == 1
    report = json.loads(paths[0].read_text())
    assert report["rows"][0]["question"] == item.question
    assert report["rows"][0]["hit"] is True
    assert report["rows"][0]["retrieved_chunks"]
    assert report["rows"][0]["trace"]["returned"] > 0
    assert recent_events(tmp_path)[0]["kind"] == "evaluation"
    enable_scoped_passage_search(app, monkeypatch)
    app.chat_input[0].set_value(item.question).run(timeout=10)
    assert not app.exception
    event = recent_events(tmp_path)[0]
    assert event["kind"] == "search" and event["returned"] > 0
    assert item.question not in json.dumps(event)


def test_ui_reranks_selected_pool(index, tmp_path, monkeypatch):
    from envsearch import rerank
    calls = []
    class Scorer:
        def score(self, question, texts):
            calls.append((question, texts))
            return list(range(len(texts)))
    monkeypatch.setattr(rerank, "get_reranker", lambda: Scorer())
    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=10)
    enable_scoped_passage_search(app, monkeypatch)
    next(c for c in app.checkbox if c.label == "Rerank candidates").check().run()
    app.chat_input[0].set_value("satellite accumulation gallons").run(timeout=10)
    assert not app.exception and calls
    assert app.session_state["messages"][-1]["trace"]["reranked_candidates"] == len(calls[0][1])


def test_broken_monitoring_does_not_block_search(index, tmp_path, monkeypatch):
    index.save(tmp_path / "index")
    (tmp_path / "monitoring.sqlite3").write_bytes(b"invalid database")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=10)
    assert not app.exception
    enable_scoped_passage_search(app, monkeypatch)
    app.chat_input[0].set_value("satellite accumulation gallons").run(timeout=10)
    assert not app.exception
    assert any("candidate passages" in m.value for m in app.markdown)
    assert any("monitoring" in w.value.lower() for w in app.warning)


def test_openai_key_choice_chat_and_evaluation(index, tmp_path, monkeypatch):
    import json
    openai = pytest.importorskip("openai")
    from envsearch import evaluate
    from tests.test_openai_chat import Stream, responding_with
    from types import SimpleNamespace as NS
    calls = []
    class Client:
        responses = NS(create=responding_with(Stream()))
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
    def create_client(**kw):
        calls.append(kw)
        return Client()
    monkeypatch.setattr(openai, "OpenAI", create_client)
    monkeypatch.setattr(evaluate, "load_items", lambda _: [evaluate.Item(
        "fixture", "satellite accumulation gallons", gold_docs=["fx-saa"], must_include=[["55"]])])
    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=10)
    next(s for s in app.selectbox if s.label == "Answer provider").select("OpenAI").run()
    next(t for t in app.text_input if t.label == "OpenAI API key").set_value("test-own-key").run()
    next(r for r in app.radio if r.label == "Response").set_value("Chat with sources").run()
    app.chat_input[0].set_value("satellite accumulation gallons").run(timeout=10)
    assert not app.exception
    assert calls[0]["api_key"] == "test-own-key"
    assert app.session_state["messages"][-1]["citations"][0].doc_id == "fx-saa"
    next(r for r in app.radio if r.label == "What would you like to check?").set_value("Check sources and answers").run()
    next(b for b in app.button if b.label == "Run test").click().run(timeout=10)
    assert not app.exception
    report = json.loads(next((tmp_path / "evals").glob("*.json")).read_text())
    assert report["config"]["provider"] == "openai"
    assert report["rows"][0]["correct"] is True
    assert report["summary"]["overall"]["retrieval.doc_recall@k"] == 1
    assert "test-own-key" not in json.dumps(report)
    assert b"test-own-key" not in (tmp_path / "monitoring.sqlite3").read_bytes()
    next(s for s in app.selectbox if s.label == "Answer provider").select("Anthropic").run()
    next(s for s in app.selectbox if s.label == "Answer provider").select("OpenAI").run()
    assert next(t for t in app.text_input if t.label == "OpenAI API key").value == "test-own-key"


@pytest.mark.parametrize("provider", ["Anthropic", "OpenAI"])
@pytest.mark.parametrize("mode", ["Chat with sources", "Find passages"])
def test_ui_scope_rejection_has_no_retrieval_or_citations(index, tmp_path, monkeypatch, provider, mode):
    from types import SimpleNamespace as NS
    import anthropic
    openai = pytest.importorskip("openai")
    from envsearch.index import Index
    from envsearch.monitoring import recent_events
    from tests.conftest import response, text_block
    calls = []
    def classify(**kw):
        calls.append(kw)
        output = '{"decision":"out_of_scope","query":""}'
        return (NS(status="completed", output_text=output) if provider == "OpenAI"
                else response(text_block(output)))
    class Client:
        messages = NS(create=classify)
        responses = NS(create=classify)
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "test-not-real")
    monkeypatch.setattr(anthropic, "Anthropic", lambda **_: Client())
    monkeypatch.setattr(openai, "OpenAI", lambda **_: Client())
    def forbidden(*_, **__):
        pytest.fail("Out-of-scope chat must not retrieve")
    monkeypatch.setattr(Index, "search", forbidden)
    app = AppTest.from_file(str(APP)).run(timeout=10)
    next(s for s in app.selectbox if s.label == "Answer provider").select(provider).run()
    next(r for r in app.radio if r.label == "Response").set_value(mode).run()
    app.chat_input[0].set_value("ผมชื่ออะไร").run(timeout=10)
    assert not app.exception and len(calls) == 1
    message = app.session_state["messages"][-1]
    assert "นอกขอบเขต" in message["content"] and message["citations"] == []
    assert message["trace"]["scope_decision"] == "out_of_scope"
    assert recent_events(tmp_path)[0]["returned"] == 0


def test_add_gold_question_and_evaluate_without_restart(index, tmp_path, monkeypatch):
    import json
    from envsearch.gold import local_path
    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=10)
    next(t for t in app.text_area if t.label == "Question to test").set_value("Satellite accumulation: what is the gallon limit?")
    next(m for m in app.multiselect if m.label == "Expected source documents").set_value(["fx-saa"])
    next(t for t in app.text_area if t.label == "Words or numbers the answer must include").set_value("55 | fifty-five\ngallons")
    next(t for t in app.text_input if t.label == "Expected pages (optional)").set_value("1")
    next(t for t in app.text_input if t.label == "Question groups / tags (optional)").set_value("en, user-added")
    next(b for b in app.button if b.label == "Save test question").click().run(timeout=10)
    assert not app.exception
    row = json.loads(local_path(tmp_path).read_text())["items"][0]
    assert row["gold_docs"] == ["fx-saa"] and row["gold_pages"] == [1]
    assert row["must_include"] == [["55", "fifty-five"], ["gallons"]]
    assert any("Saved your test question" in s.value for s in app.success)
    next(m for m in app.multiselect if m.label == "Choose individual questions (optional)").set_value([row["id"]]).run()
    next(b for b in app.button if b.label == "Run test").click().run(timeout=10)
    assert not app.exception
    report = json.loads(next((tmp_path / "evals").glob("*.json")).read_text())
    assert len(report["rows"]) == 1 and report["rows"][0]["id"] == row["id"]
    assert report["rows"][0]["hit"] is True
    # A fresh browser session sees the addition too.
    fresh = AppTest.from_file(str(APP)).run(timeout=10)
    assert any(row["id"] in option for option in next(m.options for m in fresh.multiselect if m.label == "Choose individual questions (optional)"))


def test_gold_form_validation_and_scope_case(index, tmp_path, monkeypatch):
    import json
    from envsearch.gold import local_path
    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    app = AppTest.from_file(str(APP)).run(timeout=10)
    next(t for t in app.text_area if t.label == "Question to test").set_value("ผมชื่ออะไร")
    next(b for b in app.button if b.label == "Save test question").click().run()
    assert not app.exception and not local_path(tmp_path).exists()
    assert any("source document" in e.value for e in app.error)
    next(s for s in app.selectbox if s.label == "What should the assistant do?").select("Off topic — decline without searching").run()
    next(b for b in app.button if b.label == "Save test question").click().run()
    assert not app.exception
    row = json.loads(local_path(tmp_path).read_text())["items"][0]
    assert row["question"] == "ผมชื่ออะไร" and row["expected_scope"] == "out_of_scope"
    assert row["answerable"] is False and row["gold_docs"] == []


def test_copy_latest_question_does_not_copy_generated_labels(index, tmp_path, monkeypatch):
    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    app = AppTest.from_file(str(APP))
    app.session_state["messages"] = [{"role":"user", "content":"What is the BOD limit?"},
                                     {"role":"assistant", "content":"An unverified answer"}]
    app.run(timeout=10)
    next(b for b in app.button if b.label == "Use latest conversation question").click().run()
    assert not app.exception
    assert next(t.value for t in app.text_area if t.label == "Question to test") == "What is the BOD limit?"
    assert next(t.value for t in app.text_area if t.label == "Words or numbers the answer must include") == ""
    assert next(m.value for m in app.multiselect if m.label == "Expected source documents") == []


def test_gold_question_can_preview_each_of_multiple_source_documents(index, tmp_path, monkeypatch):
    from envsearch import evaluate
    item = evaluate.Item("two-sources", "Compare storage rules and categories", gold_docs=["fx-saa", "fx-cat"],
                         must_include=[["55"]])
    monkeypatch.setattr(evaluate, "load_items", lambda _: [item])
    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    app = AppTest.from_file(str(APP)).run(timeout=10)
    assert not app.exception
    for did in item.gold_docs:
        source = app.selectbox(key=f"gold-{item.id}-{did}")
        source.select_index(len(source.options) - 1).run()
        assert not app.exception
        selected = app.selectbox(key=f"gold-{item.id}-{did}").value
        assert index.by_id[selected].doc_id == did
        assert any(m.value == index.by_id[selected].text for m in app.markdown)
