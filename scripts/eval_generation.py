"""Answer the dev questions with a generation run and check the citations of each answer."""

import argparse
import gc
import json
import sys
import time
from statistics import mean

import numpy as np

from eval_retrieval import CHUNKERS, DEPTH, QUESTIONS, RAW, RESULTS, ROOT, RUNS, SPLIT, build_retriever

from ai_act_rag.generation.context import select_sources
from ai_act_rag.generation.generate import GENERATION_OPTIONS, generate
from ai_act_rag.generation.prompt import build_messages
from ai_act_rag.generation.verify import cited_units, cites_gold, coverage, valid_sources
from ai_act_rag.ingestion.parse_eurlex import load_soup, parse_units

RESULTS = ROOT / "eval" / "generation_results"
GENERATION_RUNS = {
    "gemma3_rerank_1000w": {"model": "gemma3:12b", "retrieval": "dense_qwen3_v2_rerank", "context_words": 1000},
    "gemma3_rerank_1000w_r1": {"model": "gemma3:12b", "retrieval": "dense_qwen3_v2_rerank", "context_words": 1000},
    "gemma3_rerank_1000w_r2": {"model": "gemma3:12b", "retrieval": "dense_qwen3_v2_rerank", "context_words": 1000},
    "gemma3_rerank_2000w": {"model": "gemma3:12b", "retrieval": "dense_qwen3_v2_rerank", "context_words": 2000},
    "gemma3_no_context": {"model": "gemma3:12b", "retrieval": None},
}


def retrieve(run: dict, questions: list[dict]) -> tuple[dict, dict]:
    """Return the sources of each question and the configuration of the retrieval run."""

    if run["retrieval"] is None:
        return {question["id"]: [] for question in questions}, {}

    retrieval_run = RUNS[run["retrieval"]]
    chunks = CHUNKERS[retrieval_run["chunking"]](parse_units(load_soup(RAW)))
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    retriever, details = build_retriever(retrieval_run, chunks)

    sources = {}
    for question in questions:
        results = retriever.search(question["question"], DEPTH)
        sources[question["id"]] = select_sources(results, chunks_by_id, run["context_words"])

    # The embedding model and the reranker leave the GPU to the generator.
    del retriever
    gc.collect()
    if "torch" in sys.modules:
        sys.modules["torch"].cuda.empty_cache()
    return sources, {run["retrieval"]: retrieval_run | details}


def answer_question(model: str, question: dict, sources: list[dict]) -> dict:
    """Return the answer to one question, with the outcome of each check."""

    output = generate(model, build_messages(question["question"], [source["text"] for source in sources]))
    answer = output["answer"]
    gold = question["gold_units"]
    source_unit_ids = [source["unit_ids"] for source in sources]
    in_context = {unit_id for unit_ids in source_unit_ids for unit_id in unit_ids}

    row = {
        "id": question["id"],
        "type": question["type"],
        "question": question["question"],
        "gold_units": gold,
        "sources": [source["chunk_id"] for source in sources],
        "gold_in_context": bool(set(gold) & in_context) if gold else None,
        "answer": answer,
        **{key: value for key, value in output.items() if key != "answer"},
    }
    if answer is None:
        return row

    row["abstained"] = answer["abstain"]
    # An abstention that still makes claims is recorded, not discarded.
    row["inconsistent"] = answer["abstain"] and bool(answer["claims"])
    row["cited_units"] = cited_units(answer, source_unit_ids)
    row["checks"] = {
        "valid_sources": valid_sources(answer, len(sources)),
        "coverage": coverage(answer),
        "cites_gold": cites_gold(answer, source_unit_ids, gold),
    }
    return row


def summarise(rows: list[dict]) -> dict:
    """Return the counts of the outcomes over all the questions."""

    parsed = [row for row in rows if row["answer"] is not None]
    answerable = [row for row in parsed if row["gold_units"]]
    unanswerable = [row for row in parsed if not row["gold_units"]]
    answered = [row for row in parsed if not row["abstained"]]
    return {
        "questions": len(rows),
        "invalid_answers": len(rows) - len(parsed),
        "prompt_truncated": sum(row["prompt_truncated"] for row in rows),
        "gold_in_context": sum(bool(row["gold_in_context"]) for row in rows),
        "correct_abstentions": sum(row["abstained"] for row in unanswerable),
        "wrong_abstentions": sum(row["abstained"] for row in answerable),
        "answered_unanswerable": sum(not row["abstained"] for row in unanswerable),
        "inconsistent": sum(row["inconsistent"] for row in parsed),
        "invalid_sources": sum(not row["checks"]["valid_sources"] for row in answered),
        "uncovered": sum(not row["checks"]["coverage"] for row in answered),
        "cites_gold": sum(bool(row["checks"]["cites_gold"]) for row in answerable),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=GENERATION_RUNS)
    parser.add_argument("--split", choices=["dev", "test"], default=SPLIT)
    args = parser.parse_args()
    run_name, split = args.run, args.split
    run = GENERATION_RUNS[run_name]
    output_name = run_name if split == SPLIT else f"{run_name}_{split}"

    with open(QUESTIONS, encoding="utf-8") as file:
        questions = [json.loads(line) for line in file]
    questions = [question for question in questions if question["split"] == split]

    # All the retrieval first, then all the generation: the two never share the GPU.
    start = time.perf_counter()
    sources, retrieval_config = retrieve(run, questions)
    retrieval_seconds = time.perf_counter() - start

    rows = []
    for question in questions:
        rows.append(answer_question(run["model"], question, sources[question["id"]]))
        print(f"{question['id']} {question['type']:<11} abstained={rows[-1].get('abstained')} {rows[-1].get('checks')}")

    summary = summarise(rows)
    for key, value in summary.items():
        print(f"{key:<24}{value:>4}")

    seconds = [row["seconds"]["total"] for row in rows]
    p50, p95 = np.percentile(seconds, [50, 95])
    config = {
        "run": run_name,
        **run,
        "retrieval": retrieval_config,
        "options": GENERATION_OPTIONS,
        "split": split,
        "retrieval_seconds": round(retrieval_seconds, 2),
        "generation_seconds": {"mean": round(mean(seconds), 3), "p50": round(float(p50), 3), "p95": round(float(p95), 3)},
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"gen_{output_name}.jsonl", "w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(RESULTS / f"gen_{output_name}_summary.json", "w", encoding="utf-8") as file:
        json.dump({"config": config, "summary": summary}, file, indent=2, ensure_ascii=False)
    print(f"saved {(RESULTS / f'gen_{output_name}.jsonl').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
