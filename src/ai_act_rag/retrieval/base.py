from typing import Protocol, TypedDict


class SearchResult(TypedDict):
    chunk_id: str
    # The units the chunk comes from: a chunk can be a piece of one unit or span several.
    unit_ids: list[str]
    # Comparable only between results of the same retriever.
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]:
        """Return at most k chunks, from the highest score to the lowest."""
        ...
