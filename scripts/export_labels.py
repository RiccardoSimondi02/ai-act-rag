"""Export answers and sampled claims of the generation runs to CSV files, to label them by hand."""

import csv
import json
import random
import sys

from eval_generation import RESULTS as GENERATION_RESULTS
from eval_retrieval import QUESTIONS, RAW, ROOT

from ai_act_rag.ingestion.chunking import units_to_chunks_v2
from ai_act_rag.ingestion.parse_eurlex import load_soup, parse_units

LABELS = ROOT / "eval" / "labels"
RAG_RUN = "gemma3_rerank_1000w"
NO_CONTEXT_RUN = "gemma3_no_context"
SEED = 42
N_NO_CONTEXT = 10
N_CLAIMS = 40

ANSWER_COLUMNS = ["id", "run", "type", "question", "expected_answer", "abstained", "claims",
                  "correctness", "off_topic", "abstention_ok"]
CLAIM_COLUMNS = ["id", "claim_index", "question", "claim", "sources_text", "supported"]


def load_rows(run_name: str) -> list[dict]:
    with open(GENERATION_RESULTS / f"gen_{run_name}.jsonl", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def format_claims(answer: dict | None) -> str:
    if answer is None:
        return "(invalid answer)"
    return "\n".join(f"{index}. {claim['text']}" for index, claim in enumerate(answer["claims"], start=1))


def answer_record(row: dict, run_name: str, expected: dict[str, str]) -> dict:
    return {
        "id": row["id"],
        "run": run_name,
        "type": row["type"],
        "question": row["question"],
        "expected_answer": expected[row["id"]],
        "abstained": row.get("abstained"),
        "claims": format_claims(row["answer"]),
    }


def sources_text(row: dict, claim: dict, chunks_by_id: dict[str, dict]) -> str:
    """Return the texts of the sources cited by a claim, with the numbers the model saw."""

    parts = []
    for number in claim["sources"]:
        if 1 <= number <= len(row["sources"]):
            parts.append(f"[{number}] {chunks_by_id[row['sources'][number - 1]]['text']}")
        else:
            parts.append(f"[{number}] (not among the sources)")
    return "\n\n".join(parts)


def write_csv(path, columns: list[str], records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter=";")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    answers_path, claims_path = LABELS / "answers.csv", LABELS / "claims.csv"
    if answers_path.exists() or claims_path.exists():
        sys.exit(f"{LABELS.relative_to(ROOT)} already has label files: delete them first to export again.")

    with open(QUESTIONS, encoding="utf-8") as file:
        expected = {question["id"]: question["expected_answer"] for question in map(json.loads, file)}
    rag_rows = load_rows(RAG_RUN)
    no_context_rows = load_rows(NO_CONTEXT_RUN)

    chosen_ids = set(random.Random(SEED).sample(sorted(row["id"] for row in no_context_rows), N_NO_CONTEXT))
    answers = [answer_record(row, RAG_RUN, expected) for row in rag_rows]
    answers += [answer_record(row, NO_CONTEXT_RUN, expected) for row in no_context_rows if row["id"] in chosen_ids]

    # Every claim of the RAG answers that did not abstain, in file order so the sample is reproducible.
    all_claims = [
        (row, index, claim)
        for row in rag_rows
        if row["answer"] is not None and not row["abstained"]
        for index, claim in enumerate(row["answer"]["claims"], start=1)
    ]
    sampled = random.Random(SEED).sample(all_claims, min(N_CLAIMS, len(all_claims)))
    sampled.sort(key=lambda item: (item[0]["id"], item[1]))

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in units_to_chunks_v2(parse_units(load_soup(RAW)))}
    claims = [
        {
            "id": row["id"],
            "claim_index": index,
            "question": row["question"],
            "claim": claim["text"],
            "sources_text": sources_text(row, claim, chunks_by_id),
        }
        for row, index, claim in sampled
    ]

    LABELS.mkdir(parents=True, exist_ok=True)
    write_csv(answers_path, ANSWER_COLUMNS, answers)
    write_csv(claims_path, CLAIM_COLUMNS, claims)
    print(f"{len(answers)} answers, {len(claims)} claims sampled out of {len(all_claims)}, saved in {LABELS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
