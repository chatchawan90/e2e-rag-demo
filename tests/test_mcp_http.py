import json

from starlette.testclient import TestClient


def rpc(client, method, params=None, **kwargs):
    return client.post('/mcp', json={'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params or {}},
                       headers={'Accept': 'application/json, text/event-stream', **kwargs.pop('headers', {})}, **kwargs)


def test_public_http_tools_work_without_paid_answer_tool(index, monkeypatch):
    from envsearch import mcp_server
    from envsearch.mcp_http import create_app
    monkeypatch.setattr(mcp_server, '_index', lambda **_: index)
    with TestClient(create_app(public_url='https://demo.example.com')) as client:
        assert client.get('/healthz').status_code == 200
        result = rpc(client, 'tools/list', headers={'Host': 'demo.example.com'})
        assert result.status_code == 200
        assert {t['name'] for t in result.json()['result']['tools']} == {'search_regulations', 'get_passage', 'list_documents'}
        result = rpc(client, 'tools/call', {'name': 'search_regulations', 'arguments': {
            'query': 'satellite accumulation gallons', 'mode': 'bm25', 'k': 2}}, headers={'Host': 'demo.example.com'})
        assert '55 gallons' in json.dumps(result.json())
        bad = rpc(client, 'tools/call', {'name': 'search_regulations', 'arguments': {'query': 'x' * 6001}}, headers={'Host': 'demo.example.com'})
        assert bad.json()['result']['isError']
        paid = rpc(client, 'tools/call', {'name': 'ask_with_citations', 'arguments': {'question': 'test'}}, headers={'Host': 'demo.example.com'})
        assert paid.json()['result']['isError']


def test_http_rejects_untrusted_host_origin_and_large_requests(index, monkeypatch):
    from envsearch import mcp_server
    from envsearch.mcp_http import create_app
    monkeypatch.setattr(mcp_server, '_index', lambda **_: index)
    with TestClient(create_app(public_url='https://demo.example.com')) as client:
        assert rpc(client, 'tools/list', headers={'Host': 'evil.example'}).status_code == 421
        assert rpc(client, 'tools/list', headers={'Host': 'demo.example.com', 'Origin': 'https://evil.example'}).status_code == 403
        assert client.post('/mcp', content=b'x' * 70000, headers={'Host': 'demo.example.com', 'Content-Type': 'application/json'}).status_code == 413
