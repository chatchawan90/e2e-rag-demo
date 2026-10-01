import asyncio

from envsearch import mcp_server


def test_mcp_tools_are_registered():
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert names == {"search_regulations", "get_passage", "list_documents", "ask_with_citations"}


def test_mcp_tools_against_fixture_index(index, monkeypatch):
    monkeypatch.setattr(mcp_server, "_index", lambda **_: index)
    out = mcp_server.search_regulations("satellite accumulation gallons", k=2)
    assert "doc_id=fx-saa" in out and "page=1" in out and "#page=1" in out
    cid = out.split("chunk_id=")[1].split()[0]
    assert "55 gallons" in mcp_server.get_passage(cid)
    assert "fx-th-eff" in mcp_server.list_documents()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY" in mcp_server.ask_with_citations("anything")


def test_mcp_search_empty_result_gives_guidance(index, monkeypatch):
    monkeypatch.setattr(mcp_server, "_index", lambda **_: index)
    assert "No matching passages" in mcp_server.search_regulations("zzzz qqqq")
