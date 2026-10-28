import time

from ai_act_rag.retrieval.base import SearchResult


class RerankedRetriever:
    """Reorder the first candidates of another retriever with a cross-encoder."""

    def __init__(self, first_stage, chunks: dict[str, dict], model, n_candidates: int = 30) -> None:
        self.model = model
        self.first_stage = first_stage
        self.chunks = chunks
        self.n_candidates = n_candidates

        self.timings = []


    def search(self, query: str, k: int) -> list[SearchResult]:
        start = time.perf_counter()
        candidates = self.first_stage.search(query, max(k, self.n_candidates))
        first_stage_seconds = time.perf_counter() - start

        head = candidates[: self.n_candidates]
        tail = candidates[self.n_candidates :]
        pairs = [(query, self.chunks[candidate["chunk_id"]]["text"]) for candidate in head]

        start = time.perf_counter()
        scores = self.model.predict(pairs)
        rerank_seconds = time.perf_counter() - start
        
        ranked = sorted(zip(head, scores), key=lambda pair: pair[1], reverse=True)
        reranked = [                                                           
            {"chunk_id": candidate["chunk_id"], "unit_ids": candidate["unit_ids"], "score": float(score)}
            for candidate, score in ranked
        ]

        self.timings.append((first_stage_seconds, rerank_seconds))

        # The tail keeps the order and the scores of the first stage, not comparable with the reranker's.
        return (reranked + tail)[:k]   

