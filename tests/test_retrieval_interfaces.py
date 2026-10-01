import json

import pytest

from envsearch import cli, mcp_server
from envsearch.answer import answer, NOT_FOUND
from envsearch.evaluate import Item, run
from tests.conftest import fake_client, response, text_block, sr_citation
from tests.test_hybrid import hybrid_index, FixedEmbedder
from tests.conftest import FIX


@pytest.fixture
def saved_hybrid(hybrid_index, tmp_path, monkeypatch):
    from envsearch import vectors
    hybrid_index.save(tmp_path / "index")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.setattr(vectors, "E5Embedder", lambda model_name: FixedEmbedder())
    return hybrid_index


def test_cli_search_exposes_modes_and_intersecting_filters(saved_hybrid, capsys):
    cli.main(["search", "water quality", "--mode", "vector", "--jurisdiction", "TH",
              "--lang", "th", "--doc", "th", "--filter", "topic=effluent", "-k", "1"])
    output = capsys.readouterr().out
    assert "th p.1" in output and "น้ำทิ้ง" in output
    cli.main(["search", "water quality", "--mode", "bm25"])
    assert capsys.readouterr().out == ""


def test_cli_rejects_malformed_metadata_filter(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["search", "water", "--filter", "topic"])
    assert error.value.code == 2
    assert "KEY=VALUE" in capsys.readouterr().err


def test_answer_filters_passages_and_preserves_citations(hybrid_index):
    def respond(kw):
        passages = [b for b in kw["messages"][0]["content"] if b["type"] == "search_result"]
        assert len(passages) == 1
        assert passages[0]["source"] == "https://example.org/th.pdf#page=1"
        return response(text_block("20 mg/L", [sr_citation(0)]))
    result = answer("water quality", hybrid_index, fake_client(respond), model="m", expand=False,
                    mode="vector", jurisdiction="TH", metadata={"edition": "2024"})
    assert result.citations[0].doc_id == "th"


def test_no_matching_metadata_refuses_without_calling_model(hybrid_index):
    def fail(_):
        pytest.fail("No answer model should be called when filters exclude all documents")
    result = answer("water quality", hybrid_index, fake_client(fail), model="m", expand=False,
                    mode="hybrid", metadata={"topic": "unknown"})
    assert not result.found and result.text == NOT_FOUND


def test_mcp_search_exposes_filters_and_dense_mode(hybrid_index, monkeypatch):
    monkeypatch.setattr(mcp_server, "_index", lambda **_: hybrid_index)
    result = mcp_server.search_regulations("water quality", mode="vector", jurisdiction="TH",
                                           language="th", doc_ids=["th"], metadata={"topic": "effluent"})
    assert "doc_id=th" in result and "doc_id=us" not in result


def test_mcp_answer_respects_filters(hybrid_index, monkeypatch):
    monkeypatch.setattr(mcp_server, "_index", lambda **_: hybrid_index)
    monkeypatch.setattr(mcp_server, "get_client", lambda: fake_client(
        lambda _: response(text_block("20 mg/L", [sr_citation(0)]))))
    result = mcp_server.ask_with_citations("water quality", mode="vector", jurisdiction="TH",
                                          metadata={"edition": "2024"}, expand=False)
    assert "https://example.org/th.pdf#page=1" in result


def test_eval_uses_selected_mode_and_records_it(hybrid_index):
    items = [Item("water", "water quality", gold_docs=["th"], tags=["xl"])]
    common = dict(model="m", rewrite_model="r", retrieval_only=True, expand=False, log=lambda *_: None, k=1)
    lexical = run(items, hybrid_index, None, mode="bm25", **common)
    dense = run(items, hybrid_index, None, mode="vector", **common)
    assert lexical["summary"]["overall"]["retrieval.hit@k"] == 0
    assert dense["summary"]["overall"]["retrieval.hit@k"] == 1
    assert dense["config"]["mode"] == "vector"
    assert dense["config"]["embedding_model"] == FixedEmbedder.model_name
    json.dumps(dense)


def test_mcp_lexical_and_document_tools_work_with_stale_vectors(hybrid_index, tmp_path, monkeypatch):
    hybrid_index.save(tmp_path / "index")
    (tmp_path / "index" / "vectors.json").write_text("{}")
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    mcp_server._index.cache_clear()
    try:
        result = mcp_server.search_regulations("satellite", mode="bm25")
        assert "doc_id=us" in result
        assert "US" in mcp_server.list_documents()
        assert "satellite" in mcp_server.get_passage("us:p1:c0")
        with pytest.raises(ValueError, match="[Rr]ebuild"):
            mcp_server.search_regulations("water quality", mode="hybrid")
    finally:
        mcp_server._index.cache_clear()


def test_cli_build_vectors_then_search(tmp_path, monkeypatch, capsys):
    from envsearch import vectors
    monkeypatch.setattr(vectors, "E5Embedder", lambda model_name=None: FixedEmbedder())
    monkeypatch.setenv("ENVSEARCH_HOME", str(tmp_path))
    monkeypatch.setenv("ENVSEARCH_MANIFEST", str(FIX / "manifest.yaml"))
    cli.main(["build", "--vectors"])
    assert (tmp_path / "index" / "embeddings.npy").exists()
    capsys.readouterr()
    cli.main(["search", "water quality", "--mode", "hybrid", "--jurisdiction", "TH", "-k", "1"])
    assert "fx-th-eff p.1" in capsys.readouterr().out
