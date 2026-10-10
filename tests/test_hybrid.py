import pytest

from ai_act_rag.retrieval.hybrid import HybridRetriever


class Fixed:
    """A retriever that always returns the same ranking."""

    def __init__(self, chunk_ids: list[str]) -> None:
        self.chunk_ids = chunk_ids

    def search(self, query: str, k: int) -> list[dict]:
        return [{"chunk_id": chunk_id, "unit_ids": [chunk_id], "score": 0.0} for chunk_id in self.chunk_ids[:k]]


def test_reciprocal_rank_fusion_rewards_agreement():
    hybrid = HybridRetriever([Fixed(["x", "y", "z"]), Fixed(["y", "w"])], rrf_k=60)
    results = hybrid.search("any question", k=10)
    assert [result["chunk_id"] for result in results] == ["y", "x", "w", "z"]
    assert results[0] == {"chunk_id": "y", "unit_ids": ["y"], "score": pytest.approx(1 / 62 + 1 / 61)}
    assert len(hybrid.search("any question", k=2)) == 2


def test_fusion_of_one_ranking_keeps_its_order():
    results = HybridRetriever([Fixed(["x", "y", "z"])]).search("any question", k=10)
    assert [result["chunk_id"] for result in results] == ["x", "y", "z"]
