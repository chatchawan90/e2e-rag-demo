import math

import pytest

from envsearch.evaluate import document_metrics


def test_recall_counts_all_gold_docs_and_ndcg_rewards_order():
    result = document_metrics(["wrong", "a", "a"], ["a", "b"], k=3)
    assert result["doc_recall"] == .5
    assert result["doc_ndcg"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert document_metrics(["a", "b", "wrong"], ["a", "b"], k=3)["doc_ndcg"] == 1


def test_duplicates_do_not_inflate_doc_metrics():
    assert document_metrics(["a", "a", "a"], ["a", "b"], k=3)["doc_recall"] == .5
    assert document_metrics(["a", "a", "b"], ["a", "b"], k=2)["doc_recall"] == .5


def test_empty_retrieval_and_unlabeled_queries():
    assert document_metrics([], ["a"], k=8) == {"doc_recall": 0, "doc_ndcg": 0}
    assert document_metrics(["a"], [], k=8) == {}
