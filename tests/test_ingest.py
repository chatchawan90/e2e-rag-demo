from dataclasses import replace
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from envsearch.config import get_settings
from envsearch.index import Index
from tests.test_hybrid import FixedEmbedder


def pdf_bytes(text="The zeolite treatment process removes dissolved copper from industrial wastewater."):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    data = pdf.tobytes()
    pdf.close()
    return data


@pytest.fixture
def settings(tmp_path):
    return replace(get_settings(), home=tmp_path / "data", manifest=Path(__file__).parent / "fixtures/manifest.yaml")


def test_ingest_adds_pdf_metadata_and_preserves_existing_vectors(index, settings):
    from envsearch.ingest import ingest_pdf, uploaded_documents
    base = Index(index.chunks, index.docs)
    base.add_vectors(FixedEmbedder())
    base.save(settings.index_dir)
    previous = base.vectors.embeddings.copy()
    result = ingest_pdf(pdf_bytes(), "../copper.pdf", settings, title="Copper guidance",
                        jurisdiction="TH", metadata={"topic": "effluent"}, embedder=FixedEmbedder())
    loaded = Index.load(settings.index_dir)
    assert len(loaded.chunks) == len(base.chunks) + 1
    assert np.allclose(loaded.vectors.embeddings[:len(previous)], previous)
    hits = loaded.search(["zeolite"], mode="bm25", jurisdiction="TH", metadata={"topic": "effluent"})
    assert hits[0].chunk.doc_id == result.doc_id
    assert Path(loaded.docs[result.doc_id].path).is_relative_to(settings.home / "uploads")
    assert uploaded_documents(settings)[0].id == result.doc_id


def test_same_pdf_is_not_indexed_twice(settings):
    from envsearch.ingest import ingest_pdf
    data = pdf_bytes()
    first = ingest_pdf(data, "first.pdf", settings, with_vectors=False)
    second = ingest_pdf(data, "renamed.pdf", settings, with_vectors=False)
    assert first.doc_id == second.doc_id and second.duplicate
    assert len(Index.load(settings.index_dir).chunks) == 1


@pytest.mark.parametrize("data", [b"not a pdf", pdf_bytes("")])
def test_invalid_or_scanned_pdf_does_not_change_index(index, settings, data):
    from envsearch.ingest import ingest_pdf, uploaded_documents
    index.save(settings.index_dir)
    with pytest.raises(ValueError):
        ingest_pdf(data, "bad.pdf", settings, with_vectors=False)
    assert len(Index.load(settings.index_dir).chunks) == len(index.chunks)
    assert uploaded_documents(settings) == []


def test_embedding_failure_preserves_index_and_upload_registry(index, settings):
    from envsearch.ingest import ingest_pdf, uploaded_documents
    index.save(settings.index_dir)

    class BrokenEmbedder(FixedEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        ingest_pdf(pdf_bytes(), "new.pdf", settings, embedder=BrokenEmbedder())
    assert len(Index.load(settings.index_dir).chunks) == len(index.chunks)
    assert uploaded_documents(settings) == []


def test_cli_rebuild_keeps_uploaded_documents(settings, monkeypatch):
    from envsearch.ingest import ingest_pdf
    from envsearch.cli import main
    result = ingest_pdf(pdf_bytes(), "new.pdf", settings, with_vectors=False)
    monkeypatch.setenv("ENVSEARCH_HOME", str(settings.home))
    monkeypatch.setenv("ENVSEARCH_MANIFEST", str(settings.manifest))
    main(["build"])
    assert result.doc_id in Index.load(settings.index_dir).docs


def test_single_uploaded_pdf_is_searchable_without_vectors(settings):
    from envsearch.ingest import ingest_pdf
    result = ingest_pdf(pdf_bytes(), "new.pdf", settings, with_vectors=False)
    hits = Index.load(settings.index_dir).search(["zeolite"], mode="bm25")
    assert [h.chunk.doc_id for h in hits] == [result.doc_id]


def test_ingest_recovers_interrupted_index_before_adding_pdf(index, settings):
    from envsearch.ingest import ingest_pdf
    index.save(settings.index_dir)
    settings.index_dir.rename(settings.index_dir.with_name("index.backup"))
    result = ingest_pdf(pdf_bytes(), "new.pdf", settings, with_vectors=False)
    restored = Index.load(settings.index_dir)
    assert set(index.docs).issubset(restored.docs)
    assert result.doc_id in restored.docs


def test_failed_index_write_keeps_previous_index(index, settings, monkeypatch):
    index.save(settings.index_dir)
    def fail(*_):
        raise OSError("disk full")
    monkeypatch.setattr(Index, "_write", fail)
    with pytest.raises(OSError, match="disk full"):
        Index([], {}).save(settings.index_dir)
    assert len(Index.load(settings.index_dir).chunks) == len(index.chunks)
