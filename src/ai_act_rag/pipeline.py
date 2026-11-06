"""The question-answering pipeline of the service: retrieval, reranker, context, generation and citations."""

import re
import time
from pathlib import Path

from sentence_transformers import CrossEncoder

from ai_act_rag.generation.context import select_sources
from ai_act_rag.generation.generate import generate
from ai_act_rag.generation.prompt import build_messages
from ai_act_rag.generation.verify import valid_sources
from ai_act_rag.ingestion.chunking import units_to_chunks_v2
from ai_act_rag.ingestion.parse_eurlex import load_soup, parse_units
from ai_act_rag.retrieval.dense import DEVICE, DenseRetriever
from ai_act_rag.retrieval.rerank import RerankedRetriever

# The configuration chosen on the dev split: dense_qwen3_v2_rerank with gemma3_rerank_1000w.
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LENGTH = 512
N_CANDIDATES = 30
SEARCH_DEPTH = 100
CONTEXT_WORDS = 1000
GENERATOR_MODEL = "gemma3:12b"

# Source numbers that the model sometimes repeats inside the text, such as "[1]" or "[9, 5]".
INLINE_CITATION = re.compile(r"\s*\[\d+(?:,\s*\d+)*\]")


class Pipeline:
    """Load the models once, then answer questions with citations, stage timings and token counts."""

    def __init__(self, raw_path: Path, embeddings_dir: Path) -> None:
        units = parse_units(load_soup(raw_path))
        self.units_by_id = {unit["id"]: unit for unit in units}
        chunks = units_to_chunks_v2(units)
        self.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

        dense = DenseRetriever(chunks, EMBEDDING_MODEL, embeddings_dir)
        reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE, max_length=RERANKER_MAX_LENGTH)
        self.retriever = RerankedRetriever(dense, self.chunks_by_id, reranker, N_CANDIDATES)

    def describe_source(self, number: int, result: dict) -> dict:
        unit = self.units_by_id[result["unit_ids"][0]]
        return {
            "number": number,
            "chunk_id": result["chunk_id"],
            "citation": unit["citation"],
            "url": unit["source_url"],
            "rerank_score": round(result["score"], 4),
        }

    def answer(self, question: str) -> dict:
        start = time.perf_counter()
        results = self.retriever.search(question, SEARCH_DEPTH)
        # Taken out of the list, which would otherwise grow with every request.
        retrieval_seconds, rerank_seconds = self.retriever.timings.pop()

        chunks = select_sources(results, self.chunks_by_id, CONTEXT_WORDS)
        if not chunks:
            raise ValueError("no source fits in the context budget")
        sources = [self.describe_source(number, result) for number, result in enumerate(results[: len(chunks)], start=1)]

        output = generate(GENERATOR_MODEL, build_messages(question, [chunk["text"] for chunk in chunks]))
        answer = output["answer"]

        claims = []
        if answer is not None:
            for claim in answer["claims"]:
                numbers = [n for n in dict.fromkeys(claim["sources"]) if 1 <= n <= len(sources)]
                claims.append({
                    "text": INLINE_CITATION.sub("", claim["text"]).strip(),
                    "citations": [{key: sources[n - 1][key] for key in ("number", "citation", "url")} for n in numbers],
                })
        abstained = None if answer is None else answer["abstain"] or not claims
        if abstained:
            claims = []

        return {
            "question": question,
            "valid_answer": answer is not None,
            "abstained": abstained,
            "claims": claims,
            "sources": sources,
            "checks": {
                "valid_sources": answer is not None and valid_sources(answer, len(sources)),
                "coverage": all(claim["citations"] for claim in claims),
            },
            "top_rerank_score": sources[0]["rerank_score"],
            "tokens": {"prompt": output["prompt_tokens"], "output": output["output_tokens"]},
            "seconds": {
                "retrieval": round(retrieval_seconds, 3),
                "rerank": round(rerank_seconds, 3),
                "model_load": output["seconds"]["load"],
                "prompt": output["seconds"]["prompt"],
                "generation": output["seconds"]["output"],
                "total": round(time.perf_counter() - start, 3),
            },
        }
