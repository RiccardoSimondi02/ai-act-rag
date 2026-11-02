"""Judge the answers of a generation run with an LLM, criterion by criterion, and save the verdicts."""

import json
import sys
import time
from collections import Counter

from eval_generation import GENERATION_RUNS, RESULTS
from eval_retrieval import CHUNKERS, QUESTIONS, RAW, ROOT

from ai_act_rag.evaluation.judge import (
    JUDGE_MODEL,
    JUDGE_OPTIONS,
    judge_abstention,
    judge_claim_support,
    judge_correctness,
)
from ai_act_rag.ingestion.parse_eurlex import load_soup, parse_units


def load_chunks(retrieval_config: dict) -> dict[str, dict]:
    """Return the chunks of the retrieval run by chunk_id, or none for a run without context."""

    if not retrieval_config:
        return {}
    (retrieval_run,) = retrieval_config.values()
    chunks = CHUNKERS[retrieval_run["chunking"]](parse_units(load_soup(RAW)))
    return {chunk["chunk_id"]: chunk for chunk in chunks}


def cited_texts(row: dict, claim: dict, chunks_by_id: dict[str, dict]) -> list[str]:
    """Return the texts of the valid sources cited by a claim, with the numbers the model saw."""

    return [
        f"[{number}] {chunks_by_id[row['sources'][number - 1]]['text']}"
        for number in claim["sources"]
        if 1 <= number <= len(row["sources"])
    ]


def judge_claim(row: dict, index: int, claim: dict, chunks_by_id: dict[str, dict]) -> dict:
    texts = cited_texts(row, claim, chunks_by_id)
    if texts:
        verdict = judge_claim_support(claim["text"], texts) or {"reason": None, "supported": None}
    else:
        verdict = {"reason": "No valid source is cited.", "supported": False}
    return {"claim_index": index, "claim": claim["text"], **verdict}


def judge_row(row: dict, expected_answer: str, chunks_by_id: dict[str, dict]) -> dict:
    """Return the verdicts on one answer; a criterion that does not apply is left as None."""

    judged = {key: row[key] for key in ("id", "type", "question")}
    judged |= {"abstained": row.get("abstained"), "correctness": None, "abstention": None,
               "claims": [], "judge_failures": []}
    if row["answer"] is None:
        judged["judge_failures"].append("invalid answer")
        return judged

    claims = [claim["text"] for claim in row["answer"]["claims"]]
    answered = not row["abstained"]
    unanswerable = not row["gold_units"]
    needed = {"correctness": answered and not unanswerable, "abstention": unanswerable}

    if needed["correctness"]:
        judged["correctness"] = judge_correctness(row["question"], expected_answer, claims)
    if needed["abstention"]:
        judged["abstention"] = judge_abstention(row["question"], expected_answer, row["abstained"], claims)
    if answered and chunks_by_id:
        judged["claims"] = [
            judge_claim(row, index, claim, chunks_by_id)
            for index, claim in enumerate(row["answer"]["claims"], start=1)
        ]

    judged["judge_failures"] = [name for name, is_needed in needed.items() if is_needed and judged[name] is None]
    judged["judge_failures"] += [f"claim {c['claim_index']}" for c in judged["claims"] if c["supported"] is None]
    return judged


def rate(count: int, total: int) -> float | None:
    return round(count / total, 3) if total else None


def summarise(judged: list[dict]) -> dict:
    claims = [claim for row in judged for claim in row["claims"] if claim["supported"] is not None]
    supported = sum(claim["supported"] for claim in claims)
    verdicts = Counter(row["correctness"]["verdict"] for row in judged if row["correctness"])
    abstentions = [row for row in judged if row["abstention"]]
    abstentions_ok = sum(row["abstention"]["abstention_ok"] for row in abstentions)
    return {
        "claims_judged": len(claims),
        "claims_supported": supported,
        "supported_rate": rate(supported, len(claims)),
        "answers_graded": sum(verdicts.values()),
        "correctness": {verdict: verdicts[verdict] for verdict in ("correct", "partial", "wrong")},
        "abstain_questions": len(abstentions),
        "correct_abstentions": abstentions_ok,
        "judge_failures": sum(len(row["judge_failures"]) for row in judged),
    }


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in GENERATION_RUNS:
        sys.exit(f"Usage: python scripts/judge_generation.py RUN, where RUN is one of: {', '.join(GENERATION_RUNS)}")
    run_name = sys.argv[1]

    with open(RESULTS / f"gen_{run_name}.jsonl", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    with open(RESULTS / f"gen_{run_name}_summary.json", encoding="utf-8") as file:
        generation_config = json.load(file)["config"]
    with open(QUESTIONS, encoding="utf-8") as file:
        expected = {question["id"]: question["expected_answer"] for question in map(json.loads, file)}
    chunks_by_id = load_chunks(generation_config["retrieval"])

    start = time.perf_counter()
    judged = []
    for row in rows:
        judged.append(judge_row(row, expected[row["id"]], chunks_by_id))
        last = judged[-1]
        supported = [claim["supported"] for claim in last["claims"]]
        print(
            f"{row['id']} {row['type']:<11} correctness={(last['correctness'] or {}).get('verdict')} "
            f"abstention_ok={(last['abstention'] or {}).get('abstention_ok')} "
            f"supported={sum(s is True for s in supported)}/{len(supported)}"
        )
    seconds = time.perf_counter() - start

    summary = summarise(judged)
    for key, value in summary.items():
        print(f"{key:<22}{value}")

    config = {
        "run": run_name,
        "judge_model": JUDGE_MODEL,
        "judge_options": JUDGE_OPTIONS,
        "generation_model": generation_config["model"],
        "judge_seconds": round(seconds, 1),
    }
    with open(RESULTS / f"judged_{run_name}.jsonl", "w", encoding="utf-8", newline="\n") as file:
        for row in judged:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(RESULTS / f"judged_{run_name}_summary.json", "w", encoding="utf-8") as file:
        json.dump({"config": config, "summary": summary}, file, indent=2, ensure_ascii=False)
    print(f"saved {(RESULTS / f'judged_{run_name}.jsonl').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
