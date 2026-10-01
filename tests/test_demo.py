from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from envsearch.demo import prepare_demo
from envsearch.index import Index


def test_export_excludes_uploads_private_data_and_overwrite(index, tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'public'
    public_doc = replace(index.docs['fx-saa'], path=None, url='https://example.org/public.pdf')
    docs = {**index.docs, public_doc.id: public_doc}
    Index(index.chunks, docs).save(source / 'index')
    (source / 'gold').mkdir()
    (source / 'gold/questions.json').write_text('private labels')
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(yaml.safe_dump({'documents': [{'id': public_doc.id, 'title': public_doc.title,
                                                      'lang': 'en', 'url': public_doc.url}]}))
    result = prepare_demo(source, destination, manifest=manifest)
    assert set(result.docs) == {'fx-saa'}
    assert {c.doc_id for c in result.chunks} == {'fx-saa'}
    assert not (destination / 'gold').exists()
    assert not (destination / 'uploads').exists()
    with pytest.raises(ValueError, match='never overwritten'):
        prepare_demo(source, destination, manifest=manifest)


def test_public_browser_ignores_server_keys_and_disables_shared_writes(index, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    index.save(tmp_path / 'index')
    monkeypatch.setenv('ENVSEARCH_HOME', str(tmp_path))
    monkeypatch.setenv('ENVSEARCH_PUBLIC_DEMO', '1')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-server-key-must-not-be-used')
    monkeypatch.setenv('ENVSEARCH_MCP_PUBLIC_URL', 'https://demo.example.com')
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/envsearch/web_app.py')).run(timeout=10)
    assert not app.exception
    assert app.chat_input[0].disabled
    buttons = {b.label for b in app.button}
    assert not buttons.intersection({'Save test question', 'Add to library', 'Run test', 'Refresh activity'})
    assert any(c.value == 'https://demo.example.com/mcp' for c in app.code)
    assert not (tmp_path / 'monitoring.sqlite3').exists()
