import pytest

from ai_act_rag.evaluation.metrics import recall_at_k, reciprocal_rank

RANKED = [["a"], ["b"], ["c"], ["d"]]


def test_recall_looks_only_at_the_first_k_chunks():
    assert recall_at_k(RANKED, ["c"], k=2) == 0
    assert recall_at_k(RANKED, ["c"], k=3) == 1


def test_recall_counts_half_when_one_of_two_gold_units_is_found():
    assert recall_at_k(RANKED, ["b", "f"], k=2) == 1 / 2


def test_recall_counts_a_repeated_unit_once():
    assert recall_at_k([["a"], ["a"], ["b"]], ["a", "z"], k=3) == 1 / 2


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, ["d", "b"]) == 1 / 2
    assert reciprocal_rank(RANKED, ["z"]) == 0


def test_a_chunk_with_several_units_is_one_position():
    ranked = [["a", "b", "c"], ["d"]]
    assert reciprocal_rank(ranked, ["d"]) == 1 / 2
    assert recall_at_k(ranked, ["d"], k=2) == 1


def test_empty_gold_is_rejected():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, [], k=5)
    with pytest.raises(ValueError):
        reciprocal_rank(RANKED, [])
