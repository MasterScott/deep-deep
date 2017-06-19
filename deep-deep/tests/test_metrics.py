# -*- coding: utf-8 -*-
"""
Source: https://gist.github.com/mblondel/7337391
"""
from deepdeep.metrics import dcg_score, ndcg_score


def test_dcg():
    # Check that some rankings are better than others
    assert dcg_score([5, 3, 2], [2, 1, 0]) > dcg_score([4, 3, 2], [2, 1, 0])
    assert dcg_score([4, 3, 2], [2, 1, 0]) > dcg_score([1, 3, 2], [2, 1, 0])

    assert dcg_score([5, 3, 2], [2, 1, 0], k=2) > dcg_score([4, 3, 2], [2, 1, 0], k=2)
    assert dcg_score([4, 3, 2], [2, 1, 0], k=2) > dcg_score([1, 3, 2], [2, 1, 0], k=2)

    # Check that sample order is irrelevant
    assert dcg_score([5, 3, 2], [2, 1, 0]) == dcg_score([2, 3, 5], [0, 1, 2])
    assert dcg_score([5, 3, 2], [2, 1, 0], k=2) == dcg_score([2, 3, 5], [0, 1, 2], k=2)


def test_ndcg():
    # Perfect rankings
    assert ndcg_score([5, 3, 2], [2, 1, 0]) == 1.0
    assert ndcg_score([2, 3, 5], [0, 1, 2]) == 1.0

    assert ndcg_score([5, 3, 2], [2, 1, 0], k=2) == 1.0
    assert ndcg_score([2, 3, 5], [0, 1, 2], k=2) == 1.0
