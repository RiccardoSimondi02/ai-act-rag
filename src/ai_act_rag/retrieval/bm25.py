import math
import re
from collections.abc import Callable

from rank_bm25 import BM25Okapi
import snowballstemmer
from ai_act_rag.retrieval.base import SearchResult
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stemmer = snowballstemmer.stemmer("english")

def tokenize(text:str) -> list[str]:
    return re.findall(r"\w+", text.lower())

def tokenize_stem(text:str) -> list[str]:
    tokenized = tokenize(text)
    return stemmer.stemWords(tokenized)

def tokenize_stem_nostop(text:str) -> list[str]:
    tokenized = tokenize(text)
    tokenized_nostop = [p for p in tokenized if p not in ENGLISH_STOP_WORDS] 
    return stemmer.stemWords(tokenized_nostop)



class BM25Lucene(BM25Okapi):
    """BM25Okapi with the IDF of Lucene: always positive, and lower the more common a word is.

    BM25Okapi gives every word found in more than half of the chunks the same weight,
    a quarter of the average IDF, which can be higher than the weight of rarer words.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, chunk_count in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - chunk_count + 0.5) / (chunk_count + 0.5))


class BM25Retriever:
    def __init__(
        self,
        chunks: list[dict],
        tokenizer: Callable[[str], list[str]] = tokenize,
        index_class: type[BM25Okapi] = BM25Okapi,
    ) -> None:
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.index = index_class([self.tokenizer(chunk["text"]) for chunk in chunks])
        

    def search(self, query: str, k: int) -> list[SearchResult]:
        query_tokenized = self.tokenizer(query)
        scores = self.index.get_scores(query_tokenized)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        top_chunks = []
        for i in order:
            top_chunks.append({"chunk_id": self.chunks[i]["chunk_id"], "unit_ids": self.chunks[i]["unit_ids"], "score": float(scores[i])})

        return top_chunks

