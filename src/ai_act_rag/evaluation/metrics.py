"""Retrieval metrics. ranked_chunks holds the unit ids of each retrieved chunk, best first."""


def recall_at_k(ranked_chunks: list[list[str]], gold: list[str], k: int) -> float:
    """Return the share of gold units found in the first k chunks."""
    if not gold:
        raise ValueError("gold units are empty")
    gold_set = set(gold)
    found = set().union(*ranked_chunks[:k])
    return len(gold_set & found) / len(gold_set)


def reciprocal_rank(ranked_chunks: list[list[str]], gold: list[str]) -> float:
    """Return 1 / position of the first chunk that holds a gold unit, or 0 if none does."""
    if not gold:
        raise ValueError("gold units are empty")
    gold_set = set(gold)
    for position, unit_ids in enumerate(ranked_chunks, start=1):
        if gold_set & set(unit_ids):
            return 1 / position
    return 0.0
