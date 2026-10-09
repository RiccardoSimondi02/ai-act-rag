from ai_act_rag.retrieval.bm25 import BM25Retriever, tokenize

CHUNKS = [
    {"chunk_id": "x", "unit_ids": ["x"], "text": "the cat sleeps"},
    {"chunk_id": "y", "unit_ids": ["y"], "text": "a bird sings"},
    {"chunk_id": "z", "unit_ids": ["z"], "text": "a dog barks"},
]


def test_tokenize_splits_exact_terms_into_pieces():
    assert tokenize("10^25") == ["10", "25"]
    assert tokenize("Article 5(1)(h)") == ["article", "5", "1", "h"]


def test_search_returns_the_best_k_chunks_first():
    results = BM25Retriever(CHUNKS).search("bird", k=2)
    assert len(results) == 2
    assert set(results[0]) == {"chunk_id", "unit_ids", "score"}
    assert results[0]["chunk_id"] == "y"
    assert results[0]["score"] > results[1]["score"] == 0
