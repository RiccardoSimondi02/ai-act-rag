"""Evaluate a retriever on the dev split and save the results of each question."""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_act_rag.retrieval.rerank import RerankedRetriever
from ai_act_rag.evaluation.metrics import recall_at_budget, recall_at_k, reciprocal_rank
from ai_act_rag.ingestion.chunking import units_to_chunks, units_to_chunks_fixed, units_to_chunks_v1, units_to_chunks_v2
from ai_act_rag.ingestion.parse_eurlex import load_soup, parse_units
import numpy as np
from rank_bm25 import BM25Okapi

from ai_act_rag.retrieval.bm25 import BM25Lucene, BM25Retriever, tokenize, tokenize_stem, tokenize_stem_nostop
from ai_act_rag.retrieval.hybrid import HybridRetriever

RAW = ROOT / "data" / "raw" / "ai_act.html"
QUESTIONS = ROOT / "eval" / "dataset" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results"
EMBEDDINGS = ROOT / "data" / "embeddings"

SPLIT = "dev"
# Chunks requested per question: more than the 10 the metrics need, to see how far down a gold unit is.
DEPTH = 100

TOKENIZERS = {
    "simple": tokenize,
    "stem": tokenize_stem,
    "stem_nostop": tokenize_stem_nostop,
}
CHUNKERS = {
    "v0": units_to_chunks,
    "v1": units_to_chunks_v1,
    "fixed": units_to_chunks_fixed,
    "v2": units_to_chunks_v2,
}
BM25_INDEXES = {
    "okapi": BM25Okapi,
    "lucene": BM25Lucene,
}
RUNS = {
    "bm25_v0": {"retriever": "bm25", "tokenizer": "simple", "chunking": "v0"},
    "bm25_stem_v0": {"retriever": "bm25", "tokenizer": "stem", "chunking": "v0"},
    "bm25_stem_v1": {"retriever": "bm25", "tokenizer": "stem", "chunking": "v1"},
    "dense_qwen3_v1": {"retriever": "dense", "model": "Qwen/Qwen3-Embedding-0.6B", "chunking": "v1"},
    "bm25_stem_nostop_v1": {"retriever": "bm25", "tokenizer": "stem_nostop", "chunking": "v1"},
    "bm25_stem_lucene_v1": {"retriever": "bm25", "tokenizer": "stem", "idf": "lucene", "chunking": "v1"},
    "hybrid_rrf_v1": {"retriever": "hybrid","parts": ["bm25_stem_v1", "dense_qwen3_v1"],"rrf_k": 60,"chunking": "v1",},
    "bm25_stem_fixed": {"retriever": "bm25", "tokenizer": "stem", "chunking": "fixed"},
    "dense_qwen3_fixed": {"retriever": "dense", "model": "Qwen/Qwen3-Embedding-0.6B", "chunking": "fixed"},
    "hybrid_rrf_fixed": {"retriever": "hybrid","parts": ["bm25_stem_fixed", "dense_qwen3_fixed"],"rrf_k": 60,"chunking": "fixed",},
    "bm25_stem_v2": {"retriever": "bm25", "tokenizer": "stem", "chunking": "v2"},
    "dense_qwen3_v2": {"retriever": "dense", "model": "Qwen/Qwen3-Embedding-0.6B", "chunking": "v2"},
    "hybrid_rrf_v2": {"retriever": "hybrid", "parts": ["bm25_stem_v2", "dense_qwen3_v2"], "rrf_k": 60, "chunking": "v2"},
    "dense_qwen3_v2_rerank": {
        "retriever": "rerank",
        "first_stage": "dense_qwen3_v2",
        "model": "BAAI/bge-reranker-v2-m3",
        "max_length": 512,
        "n_candidates": 30,
        "chunking": "v2",
    },
    "hybrid_rrf_v2_rerank": {
        "retriever": "rerank",
        "first_stage": "hybrid_rrf_v2",
        "model": "BAAI/bge-reranker-v2-m3",
        "max_length": 512,
        "n_candidates": 30,
        "chunking": "v2",
    },
}


def build_retriever(run: dict, chunks: list[dict]) -> tuple:
    """Return the retriever of a run and the details to save with its results."""

    if run["retriever"] == "bm25":
        # A BM25 run that does not name its IDF uses the one of the library.
        idf = run.get("idf", "okapi")
        retriever = BM25Retriever(chunks, TOKENIZERS[run["tokenizer"]], BM25_INDEXES[idf])
        details = {"library": "rank_bm25.BM25Okapi", "idf": idf, "k1": retriever.index.k1, "b": retriever.index.b}
        return retriever, details

    if run["retriever"] == "hybrid":
        retrievers = []
        parts = {}
        for name in run["parts"]:
            part = RUNS[name]
            # The rankings are merged by chunk id: every part must work on the same chunks.
            if part["chunking"] != run["chunking"]:
                sys.exit(f"{name} uses chunking {part['chunking']}, the hybrid run uses {run['chunking']}")
            retriever, details = build_retriever(part, chunks)
            retrievers.append(retriever)
            parts[name] = part | details
        # Saved in place of the list of names: the full configuration of each part.
        return HybridRetriever(retrievers, run["rrf_k"], DEPTH), {"parts": parts}

    if run["retriever"] == "rerank":
        name = run["first_stage"]
        first_stage_run = RUNS[name]
        if first_stage_run["chunking"] != run["chunking"]:
            sys.exit(f"{name} uses chunking {first_stage_run['chunking']}, the rerank run uses {run['chunking']}")
        first_stage, first_stage_details = build_retriever(first_stage_run, chunks)

        from sentence_transformers import CrossEncoder

        from ai_act_rag.retrieval.dense import DEVICE

        model = CrossEncoder(run["model"], device=DEVICE, max_length=run["max_length"])
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        retriever = RerankedRetriever(first_stage, chunks_by_id, model, run["n_candidates"])
        return retriever, {"first_stage": {name: first_stage_run | first_stage_details}, "device": DEVICE}


    # Imported here: loading torch takes seconds and the BM25 runs do not need it.
    from ai_act_rag.retrieval.dense import DenseRetriever

    retriever = DenseRetriever(chunks, run["model"], EMBEDDINGS)
    details = {
        "library": "sentence-transformers",
        "device": str(retriever.model.device),
        "weights_dtype": str(next(retriever.model.parameters()).dtype),
        "encode_seconds": retriever.encode_seconds,
        "embeddings_from_cache": retriever.from_cache,
    }
    return retriever, details


def percentiles(seconds: list[float]) -> dict[str, float]:
    """Return the median and the 95th percentile of a list of durations."""

    p50, p95 = np.percentile(seconds, [50, 95])
    return {"p50": round(float(p50), 4), "p95": round(float(p95), 4)}


def gold_position(results: list[dict], unit_id: str) -> int | None:
    """Return the position of the first chunk that holds the unit, or None if it is not there."""

    positions = (position for position, result in enumerate(results, start=1) if unit_id in result["unit_ids"])
    return next(positions, None)


def evaluate(retriever, question: dict, words_per_chunk: dict[str, int]) -> dict:
    """Return what the retriever found for one question."""

    results = retriever.search(question["question"], DEPTH)
    gold = question["gold_units"]
    row = {
        "id": question["id"],
        "type": question["type"],
        "question": question["question"],
        "gold_positions": {unit_id: gold_position(results, unit_id) for unit_id in gold},
        # Five decimals: reciprocal rank fusion scores are at most 2 / 61.
        "top_score": round(results[0]["score"], 5),
        "top_results": [
            {"chunk_id": result["chunk_id"], "score": round(result["score"], 5)} for result in results[:5]
        ],
    }
    # Abstain questions have no gold units: the search is saved, the metrics are not defined.
    if gold:
        ranked = [result["unit_ids"] for result in results]
        chunk_words = [words_per_chunk[result["chunk_id"]] for result in results]
        row["recall@5"] = recall_at_k(ranked, gold, 5)
        row["recall@10"] = recall_at_k(ranked, gold, 10)
        row["recall@20"] = recall_at_k(ranked, gold, 20)
        row["recall@1000w"] = recall_at_budget(ranked, chunk_words, gold, 1000)
        row["recall@2000w"] = recall_at_budget(ranked, chunk_words, gold, 2000)
        row["rr"] = reciprocal_rank(ranked, gold)
    return row


def summarise(rows: list[dict]) -> dict:
    """Return the mean of each metric by question type and in total."""

    scored = [row for row in rows if "rr" in row]
    groups = defaultdict(list)
    for row in scored:
        groups[row["type"]].append(row)
    groups["total"] = scored

    return {
        name: {
            "questions": len(group),
            "recall@5": mean(row["recall@5"] for row in group),
            "recall@10": mean(row["recall@10"] for row in group),
            "recall@20": mean(row["recall@20"] for row in group),
            "recall@1000w": mean(row["recall@1000w"] for row in group),
            "recall@2000w": mean(row["recall@2000w"] for row in group),
            "mrr": mean(row["rr"] for row in group),
        }
        for name, group in groups.items()
    }


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in RUNS:
        sys.exit(f"Usage: python scripts/eval_retrieval.py RUN, where RUN is one of: {', '.join(RUNS)}")
    run_name = sys.argv[1]
    run = RUNS[run_name]

    chunks = CHUNKERS[run["chunking"]](parse_units(load_soup(RAW)))
    words_per_chunk = {chunk["chunk_id"]: len(chunk["text"].split()) for chunk in chunks}
    start = time.perf_counter()
    retriever, details = build_retriever(run, chunks)
    build_seconds = time.perf_counter() - start

    with open(QUESTIONS, encoding="utf-8") as file:
        questions = [json.loads(line) for line in file]
    questions = [question for question in questions if question["split"] == SPLIT]

    retriever.search("warm-up question", DEPTH)
    # The reranker records the time of each stage: the warm-up must not count there either.
    if hasattr(retriever, "timings"):
        retriever.timings.clear()

    rows = []
    query_seconds = []
    for question in questions:
        start = time.perf_counter()
        rows.append(evaluate(retriever, question, words_per_chunk))
        query_seconds.append(time.perf_counter() - start)
    summary = summarise(rows)

    columns = ["recall@5", "recall@10", "recall@20", "recall@1000w", "recall@2000w", "mrr"]
    print(f"{'type':<12}{'n':>4}" + "".join(f"{column:>14}" for column in columns))
    for name, values in summary.items():
        print(f"{name:<12}{values['questions']:>4}" + "".join(f"{values[column]:>14.3f}" for column in columns))
    abstain = len(rows) - summary["total"]["questions"]
    print(f"{'abstain':<12}{abstain:>4}" + "".join(f"{'-':>14}" for _ in columns))

    config = {
        "run": run_name,
        **run,
        **details,
        "split": SPLIT,
        "depth": DEPTH,
        "chunks": len(chunks),
        # Building includes loading the models; the mean per question excludes the warm-up search.
        "build_seconds": round(build_seconds, 2),
        "mean_query_seconds": round(mean(query_seconds), 4),
        "query_seconds": percentiles(query_seconds),
    }
    if hasattr(retriever, "timings"):
        config["first_stage_seconds"] = percentiles([first for first, _ in retriever.timings])
        config["rerank_seconds"] = percentiles([second for _, second in retriever.timings])
    out =RESULTS / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump({"config": config, "summary": summary, "questions": rows}, file, indent=2, ensure_ascii=False)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
