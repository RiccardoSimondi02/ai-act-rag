"""Evaluate a retriever on the dev split and save the results of each question."""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_act_rag.evaluation.metrics import recall_at_k, reciprocal_rank
from ai_act_rag.ingestion.chunking import units_to_chunks, units_to_chunks_v1
from ai_act_rag.ingestion.parse_eurlex import load_soup, parse_units
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
    "hybrid_rrf_v1": {
        "retriever": "hybrid",
        "parts": ["bm25_stem_v1", "dense_qwen3_v1"],
        "rrf_k": 60,
        "chunking": "v1",
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

    # Imported here: loading torch takes seconds and the BM25 runs do not need it.
    from ai_act_rag.retrieval.dense import DenseRetriever

    retriever = DenseRetriever(chunks, run["model"], EMBEDDINGS)
    details = {
        "library": "sentence-transformers",
        "weights_dtype": str(next(retriever.model.parameters()).dtype),
        "encode_seconds": retriever.encode_seconds,
        "embeddings_from_cache": retriever.from_cache,
    }
    return retriever, details


def gold_position(results: list[dict], unit_id: str) -> int | None:
    """Return the position of the first chunk that holds the unit, or None if it is not there."""

    positions = (position for position, result in enumerate(results, start=1) if unit_id in result["unit_ids"])
    return next(positions, None)


def evaluate(retriever, question: dict) -> dict:
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
        row["recall@5"] = recall_at_k(ranked, gold, 5)
        row["recall@10"] = recall_at_k(ranked, gold, 10)
        row["recall@20"] = recall_at_k(ranked, gold, 20)
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
    start = time.perf_counter()
    retriever, details = build_retriever(run, chunks)
    build_seconds = time.perf_counter() - start

    with open(QUESTIONS, encoding="utf-8") as file:
        questions = [json.loads(line) for line in file]
    questions = [question for question in questions if question["split"] == SPLIT]

    start = time.perf_counter()
    rows = [evaluate(retriever, question) for question in questions]
    query_seconds = (time.perf_counter() - start) / len(questions)
    summary = summarise(rows)

    print(f"{'type':<12}{'n':>4}{'recall@5':>10}{'recall@10':>11}{'recall@20':>11}{'mrr':>8}")
    for name, values in summary.items():
        print(
            f"{name:<12}{values['questions']:>4}{values['recall@5']:>10.3f}"
            f"{values['recall@10']:>11.3f}{values['recall@20']:>11.3f}{values['mrr']:>8.3f}"
        )
    abstain = len(rows) - summary["total"]["questions"]
    print(f"{'abstain':<12}{abstain:>4}{'-':>10}{'-':>11}{'-':>8}")

    config = {
        "run": run_name,
        **run,
        **details,
        "split": SPLIT,
        "depth": DEPTH,
        "chunks": len(chunks),
        # Building includes loading the model; the first question includes warming up the GPU.
        "build_seconds": round(build_seconds, 2),
        "mean_query_seconds": round(query_seconds, 4),
    }
    out =RESULTS / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump({"config": config, "summary": summary, "questions": rows}, file, indent=2, ensure_ascii=False)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
