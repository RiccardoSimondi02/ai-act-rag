import re
from rank_bm25 import BM25Okapi
from ai_act_rag.retrieval.base import SearchResult


def tokenize(text:str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    def search(self, query: str, k: int) -> list[SearchResult]:
        query_tokenized = tokenize(query)
        scores = self.index.get_scores(query_tokenized)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        top_chunks = []
        for i in order:
            top_chunks.append({"chunk_id": self.chunks[i]["chunk_id"], "unit_ids": self.chunks[i]["unit_ids"], "score": float(scores[i])})

        return top_chunks

