"""Retrieval contract tests use fixed embeddings; no model download or API calls."""
import json
from dataclasses import replace

import numpy as np
import pytest

from envsearch.corpus import Chunk, Document, load_manifest
from envsearch.index import Index
from envsearch.textnorm import tokenize


class FixedEmbedder:
    model_name = "intfloat/multilingual-e5-small"

    def embed_documents(self, texts):
        return np.array([[1, 0] if "น้ำ" in text else [0, 1] for text in texts], dtype=np.float32)

    def embed_queries(self, texts):
        return np.array([[1, 0] for _ in texts], dtype=np.float32)


@pytest.fixture
def hybrid_index():
    docs = {
        "us": Document("us", "US", "https://example.org/us.pdf", jurisdiction="US",
                       metadata={"topic": "waste", "edition": "2024"}),
        "th": Document("th", "Thai", "https://example.org/th.pdf", lang="th", jurisdiction="TH",
                       metadata={"topic": "effluent", "edition": "2024"}),
        "en-th": Document("en-th", "Translation", "https://example.org/en-th.pdf", jurisdiction="TH",
                          metadata={"topic": "effluent", "edition": "2023"}),
    }
    texts = ["satellite accumulation gallons", "น้ำทิ้ง บีโอดี 20", "factory discharge standard"]
    chunks = [Chunk(f"{did}:p1:c0", did, 1, text, docs[did].lang, tokenize(text))
              for did, text in zip(docs, texts)]
    idx = Index(chunks, docs)
    idx.add_vectors(embedder=FixedEmbedder())
    return idx


def test_vector_search_retrieves_without_shared_keywords(hybrid_index):
    assert hybrid_index.search(["water quality"], mode="bm25") == []
    hit = hybrid_index.search(["water quality"], mode="vector", k=1)[0]
    assert hit.chunk.doc_id == "th"
    assert hybrid_index.cite_url(hit.chunk) == "https://example.org/th.pdf#page=1"


def test_hybrid_fuses_lexical_and_dense_ranks(hybrid_index):
    hits = hybrid_index.search(["satellite accumulation"], mode="hybrid", k=3)
    # US ranks first in BM25 and second in vectors; Thai ranks first only in vectors.
    assert [h.chunk.doc_id for h in hits] == ["us", "th", "en-th"]
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert hybrid_index.search(["satellite accumulation"], k=3) == hits


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_filters_intersect_without_equating_language_and_jurisdiction(hybrid_index, mode):
    hits = hybrid_index.search(["factory discharge"], mode=mode, lang="en", jurisdiction="th",
                               doc_ids={"us", "en-th"}, metadata={"topic": "effluent", "edition": "2023"})
    assert [h.chunk.doc_id for h in hits] == ["en-th"]
    assert hybrid_index.search(["factory discharge"], mode=mode, metadata={"absent": "x"}) == []
    assert hybrid_index.search(["factory discharge"], mode=mode, doc_ids=set()) == []


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_filters_apply_before_candidate_limit(hybrid_index, mode):
    # More than the 20-candidate retrieval pool would otherwise hide the allowed document.
    chunks = [replace(hybrid_index.chunks[0], id=f"us:p1:c{i}") for i in range(25)]
    chunks.append(replace(hybrid_index.chunks[0], id="en-th:p1:c0", doc_id="en-th"))
    chunks.extend(replace(hybrid_index.chunks[1], id=f"us:p2:c{i}", doc_id="us") for i in range(30))
    idx = Index(chunks, hybrid_index.docs)
    idx.add_vectors(embedder=FixedEmbedder())
    hits = idx.search(["satellite"], k=1, mode=mode, jurisdiction="TH")
    assert [h.chunk.doc_id for h in hits] == ["en-th"]


def test_vectors_and_metadata_round_trip(hybrid_index, tmp_path, monkeypatch):
    from envsearch import vectors
    hybrid_index.save(tmp_path)
    monkeypatch.setattr(vectors, "E5Embedder", lambda model_name: FixedEmbedder())
    loaded = Index.load(tmp_path)
    assert loaded.docs["th"].metadata == {"topic": "effluent", "edition": "2024"}
    assert loaded.search(["water quality"], mode="vector", k=1)[0].chunk.doc_id == "th"
    assert (tmp_path / "embeddings.npy").exists()


@pytest.mark.parametrize("change", ["text", "order"])
def test_stale_vectors_are_rejected(hybrid_index, tmp_path, change):
    hybrid_index.save(tmp_path)
    path = tmp_path / "chunks.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "text":
        rows[0]["text"] = "changed source"
    else:
        rows.reverse()
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="[Rr]ebuild"):
        Index.load(tmp_path)
    assert Index.load(tmp_path, load_vectors=False).search(["satellite"], mode="bm25")


def test_bm25_rebuild_removes_old_vectors(hybrid_index, tmp_path):
    hybrid_index.save(tmp_path)
    Index(hybrid_index.chunks, hybrid_index.docs).save(tmp_path)
    loaded = Index.load(tmp_path)
    assert loaded.search(["satellite"], mode="bm25")
    with pytest.raises(ValueError, match="build --vectors"):
        loaded.search(["water quality"], mode="hybrid")


@pytest.mark.parametrize("queries,k", [([], 8), (["  "], 8), (["water quality"], 0)])
def test_empty_search_returns_no_hits(hybrid_index, queries, k):
    assert hybrid_index.search(queries, k=k) == []


def test_legacy_index_stays_bm25(index):
    assert index.search(["satellite"])
    with pytest.raises(ValueError, match="build --vectors"):
        index.search(["water quality"], mode="vector")
    with pytest.raises(ValueError, match="mode"):
        index.search(["satellite"], mode="typo")


def test_manifest_accepts_string_metadata_and_rejects_typed_values(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text('documents:\n- id: d\n  title: D\n  url: https://example.org\n  metadata: {topic: effluent, year: "2024"}\n')
    assert load_manifest(manifest)[0].metadata["year"] == "2024"
    manifest.write_text(manifest.read_text().replace('"2024"', '2024'))
    with pytest.raises(ValueError, match="metadata"):
        load_manifest(manifest)


@pytest.mark.parametrize("damage", ["missing", "shape", "nan"])
def test_damaged_vector_artifacts_have_rebuild_guidance(hybrid_index, tmp_path, damage):
    hybrid_index.save(tmp_path)
    path = tmp_path / "embeddings.npy"
    if damage == "missing":
        path.unlink()
    else:
        np.save(path, np.ones((1, 2)) if damage == "shape" else np.full((3, 2), np.nan))
    with pytest.raises(ValueError, match="[Rr]ebuild"):
        Index.load(tmp_path)


def test_dense_results_obey_document_diversity_cap(hybrid_index):
    chunks = [replace(hybrid_index.chunks[1], id=f"th:p1:c{i}") for i in range(6)]
    chunks.append(hybrid_index.chunks[0])
    idx = Index(chunks, hybrid_index.docs)
    idx.add_vectors(embedder=FixedEmbedder())
    hits = idx.search(["water quality"], mode="vector", k=8)
    assert [h.chunk.doc_id for h in hits] == ["th", "th", "th", "us"]


def test_empty_corpus_build_has_actionable_error():
    with pytest.raises(ValueError, match="empty corpus"):
        Index([], {}).add_vectors(embedder=FixedEmbedder())


def test_e5_uses_distinct_prefixes_and_normalized_text():
    from envsearch.vectors import E5Embedder

    class Encoder:
        def encode(self, texts, **kwargs):
            # Return distinct values only for the model's expected inputs.
            assert kwargs["normalize_embeddings"] and kwargs["convert_to_numpy"]
            mapping = {"query: บีโอดี 20": [1, 0], "passage: น้ำทิ้ง": [0, 1]}
            return np.array([mapping[text] for text in texts])

    embedder = E5Embedder()
    embedder._model = Encoder()  # isolate external inference; exercise the real adapter
    assert embedder.embed_queries(["บีโอดี ๒๐"]).tolist() == [[1, 0]]
    assert embedder.embed_documents(["นํ้าทิ้ง"]).tolist() == [[0, 1]]
