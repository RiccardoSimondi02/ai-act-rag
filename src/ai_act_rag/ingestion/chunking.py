
from collections import Counter


def units_to_chunks(units):
    """ Return chunks created from units"""

    chunks = []
    for unit in units:
        chunk = {"chunk_id": unit["id"], "unit_ids": [unit["id"]], "text": unit["text"] }
        chunks.append(chunk)
    return chunks

def units_to_chunks_v1(units):
    """ Return chunks created from units, the text contains also the citation"""

    chunks = []
    for unit in units:
        chunk = {"chunk_id": unit["id"], "unit_ids": [unit["id"]], "text": f"{unit["citation"]} - {unit["text"]}" }
        chunks.append(chunk)
    return chunks

def split_into_windows(items: list, size: int, overlap: int) -> list[list]:
    """Return overlapping slices of items; every item ends up in at least one slice."""

    return [items[start : start + size] for start in range(0, len(items) - overlap, size - overlap)]


def units_to_chunks_fixed(units, size=200, overlap=50):
    """ Return fixed chunks """

    chunks = []
    pairs = [(word, unit["id"]) for unit in units for word in unit["text"].split()]
    lengths = Counter(unit_id for _, unit_id in pairs)

    for window in split_into_windows(pairs, size, overlap):
        window_words_per_unit = Counter(unit_id for _, unit_id in window)
        unit_ids = [unit_id for unit_id, words in window_words_per_unit.items() if words >= 50 or 2 * words >= lengths[unit_id]]
        chunk = {"chunk_id": f"fixed_{len(chunks)}", "unit_ids": unit_ids, "text": " ".join(word for word, _ in window)}
        chunks.append(chunk)
    return chunks


def units_to_chunks_v2(units: list[dict], size: int = 200, overlap: int = 50) -> list[dict]:
    """Return v1 chunks, with units longer than size words split into windows that keep the citation."""

    chunks = []
    for unit in units:
        words = unit["text"].split()
        if len(words) <= size:
            chunks.extend(units_to_chunks_v1([unit]))
            continue
        for i, window in enumerate(split_into_windows(words, size, overlap)):
            chunk = {
                "chunk_id": f"{unit['id']}#{i}",
                "unit_ids": [unit["id"]],
                "text": f"{unit['citation']} - {' '.join(window)}",
            }
            chunks.append(chunk)
    return chunks
