from ai_act_rag.retrieval.rerank import RerankedRetriever

CHUNKS = {chunk_id: {"chunk_id": chunk_id, "unit_ids": [chunk_id], "text": chunk_id} for chunk_id in "abcd"}


class Fixed:
    """A retriever that always returns the same ranking."""

    def __init__(self, chunk_ids: list[str]) -> None:
        self.chunk_ids = chunk_ids

    def search(self, query: str, k: int) -> list[dict]:
        return [{"chunk_id": chunk_id, "unit_ids": [chunk_id], "score": 0.0} for chunk_id in self.chunk_ids[:k]]


class FakeModel:
    """A reranker that gives each text a fixed score."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.scores[text] for _, text in pairs]


def test_rerank_reorders_the_candidates_and_keeps_the_tail():
    # No score for "d": it is outside the candidates, so the model must never see it.
    model = FakeModel({"a": 0.1, "b": 0.9, "c": 0.5})
    reranker = RerankedRetriever(Fixed(["a", "b", "c", "d"]), CHUNKS, model, n_candidates=3)

    results = reranker.search("any question", k=10)
    assert [result["chunk_id"] for result in results] == ["b", "c", "a", "d"]
    assert results[0] == {"chunk_id": "b", "unit_ids": ["b"], "score": 0.9}

    assert [result["chunk_id"] for result in reranker.search("any question", k=2)] == ["b", "c"]
    assert len(reranker.timings) == 2
