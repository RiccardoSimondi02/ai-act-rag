from ai_act_rag.retrieval.base import Retriever, SearchResult


class HybridRetriever:
    """Merge the rankings of several retrievers with reciprocal rank fusion.

    Each retriever gives a chunk 1 / (rrf_k + position), and the contributions are added up.
    Positions are used instead of scores because the scores of different retrievers
    are not on the same scale.
    """

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, depth: int = 100) -> None:
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        
        self.depth = depth

    def search(self, query: str, k: int) -> list[SearchResult]:
        scores = {}
        unit_ids = {}
        for retriever in self.retrievers:
            results = retriever.search(query, self.depth)
            for position, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (self.rrf_k + position)
                unit_ids[chunk_id] = result["unit_ids"]

        order = sorted(scores, key=scores.get, reverse=True)[:k]
        return [
            {"chunk_id": chunk_id, "unit_ids": unit_ids[chunk_id], "score": scores[chunk_id]} for chunk_id in order
        ]
