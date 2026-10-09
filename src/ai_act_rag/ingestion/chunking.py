
def units_to_chunks(units):
    """ Return chunks created from units"""

    chunks = []
    for unit in units:
        chunk = {"chunk_id": unit["id"], "unit_ids": [unit["id"]], "text": unit["text"] }
        chunks.append(chunk)
    return chunks