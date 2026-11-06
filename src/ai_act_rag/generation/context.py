def select_sources(results: list[dict], chunks_by_id: dict[str, dict], budget: int) -> list[dict]:
    """Return the chunks of the first results that fit in a budget of words, like recall_at_budget."""

    selected = []
    total_words = 0
    for result in results:
        chunk = chunks_by_id[result["chunk_id"]]
        words = len(chunk["text"].split())
        if total_words + words > budget:
            break
        total_words += words
        selected.append(chunk)
    return selected
