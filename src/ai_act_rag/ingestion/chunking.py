
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