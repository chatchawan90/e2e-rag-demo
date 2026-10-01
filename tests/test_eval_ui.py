from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src/envsearch/web_app.py"


def test_evaluation_has_steps_help_and_honest_source_only_results(index, tmp_path, monkeypatch):
    from envsearch import evaluate
    monkeypatch.setattr(evaluate, "load_items", lambda _: [evaluate.Item(
        "case", "satellite accumulation gallons", gold_docs=["fx-saa"], must_include=[["55"]])])
    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=10)
    assert not app.exception
    headings = [h.value for h in app.subheader]
    assert all(label in headings for label in ("1. Choose questions", "2. Choose what to check", "3. Run test", "Results"))
    radio = next(r for r in app.radio if r.label == "What would you like to check?")
    assert radio.value == "Find the right sources" and radio.proto.help
    groups = next(m for m in app.multiselect if m.label == "Question groups (optional)")
    assert "tags" in groups.proto.help.lower()
    next(b for b in app.button if b.label == "Run test").click().run(timeout=10)
    assert not app.exception
    score = next(m for m in app.metric if m.label == "Expected source found")
    assert score.value == "100%" and score.proto.help
    assert next(m for m in app.metric if m.label == "Answer checks passed").value == "Not checked"
    assert any("Source found; answer not checked" in str(d.value.to_dict()) for d in app.dataframe)


def test_answer_test_explains_missing_key(index, tmp_path, monkeypatch):
    index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=10)
    next(r for r in app.radio if r.label == "What would you like to check?").set_value("Check sources and answers").run()
    assert not app.exception
    assert next(b for b in app.button if b.label == "Run test").disabled
    assert any("API key" in i.value for i in app.info)
