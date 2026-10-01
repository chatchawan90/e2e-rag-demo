from pathlib import Path
import json
from types import SimpleNamespace as NS

import pytest

from envsearch.corpus import load_manifest
from envsearch.index import Index

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def index(tmp_path_factory):
    docs = load_manifest(FIX / "manifest.yaml")
    idx = Index.build(docs, tmp_path_factory.mktemp("raw"), log=lambda *_: None)
    out = tmp_path_factory.mktemp("index")
    idx.save(out)
    return Index.load(out)  # round-trip through disk, like production


class FakeMessages:
    """Stands in for client.messages. `script` maps a predicate on the request to a response."""

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        if "scope router" in kw.get("system", ""):
            return scope_response(kw)
        return self.respond(kw)


def fake_client(respond):
    return NS(messages=FakeMessages(respond))


def text_block(text, citations=None):
    return NS(type="text", text=text, citations=citations)


def sr_citation(i, start=0, end=1, cited="..."):
    return NS(type="search_result_location", search_result_index=i, start_block_index=start,
              end_block_index=end, cited_text=cited, source="s", title="t")


def response(*blocks):
    return NS(content=list(blocks), usage=NS(input_tokens=100, output_tokens=20))


def scope_response(kwargs):
    """Default in-scope provider reply for tests of downstream answer/retrieval behavior."""
    text = scope_json(kwargs)
    return response(text_block(text))


def scope_json(kwargs):
    payload = json.loads(kwargs.get("messages", kwargs.get("input"))[0]["content"])
    return json.dumps({"decision": "in_scope", "query": payload["question"]})
