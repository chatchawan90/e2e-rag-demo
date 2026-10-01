from dataclasses import replace

from envsearch.index import Index
from tests.test_hybrid import hybrid_index


def test_reranker_sees_candidates_before_final_cutoff_and_diversity(hybrid_index):
    chunks = [replace(hybrid_index.chunks[0], id=f"us:p1:c{i}") for i in range(5)]
    chunks += [hybrid_index.chunks[1]]
    idx = Index(chunks, hybrid_index.docs)
    from tests.test_hybrid import FixedEmbedder
    idx.add_vectors(FixedEmbedder())

    class Reranker:
        def score(self, question, texts):
            assert question == "satellite" and len(texts) == 6
            return [100 if "น้ำ" in text else i for i, text in enumerate(texts)]

    trace = {}
    hits = idx.search(["satellite"], k=2, mode="hybrid", rerank=True, reranker=Reranker(),
                      candidate_k=10, per_doc_cap=1, trace=trace)
    assert [hit.chunk.doc_id for hit in hits] == ["th", "us"]
    assert trace["reranked_candidates"] == 6 and trace["returned"] == 2


def test_reranker_only_receives_filtered_passages(hybrid_index):
    class Reranker:
        def score(self, question, texts):
            assert texts == ["น้ำทิ้ง บีโอดี 20"]
            return [0.9]
    hits = hybrid_index.search(["water"], mode="vector", k=1, jurisdiction="TH", lang="th",
                               rerank=True, reranker=Reranker(), candidate_k=10)
    assert hits[0].chunk.doc_id == "th"


def test_trace_explains_candidate_cutoff(hybrid_index):
    trace = {}
    hits = hybrid_index.search(["water"], mode="vector", k=1, candidate_k=2, trace=trace)
    assert len(hits) == 1
    assert trace["eligible_chunks"] == 3 and trace["candidate_k"] == 2
    assert trace["fused_candidates"] == 2
