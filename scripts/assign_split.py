"""Assign each question without a split to dev or test. Assigned questions are never changed."""

import json
import random
import sys
from pathlib import Path

SEED = 42
DEV_SHARE = 0.5

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "dataset" / "questions.jsonl"


def find_neighbours(question: dict, questions: list[dict]) -> list[dict]:
    """Return the other questions that share at least one gold unit with this one."""

    gold = set(question["gold_units"])
    
    return [other for other in questions if other is not question and gold & set(other["gold_units"])]


def pick_split(question: dict, questions: list[dict]) -> str:
    """Return the split for a question that has none yet."""

    neighbours = find_neighbours(question, questions)
    splits = {other["split"] for other in neighbours if other["split"]}
    # A gold unit on both sides would leak from dev to test: there is no right answer.
    if len(splits) > 1:
        sys.exit(f"{question['id']} shares gold units with both splits")
    if splits:
        return splits.pop()
    if neighbours:
        return "dev"

    same_type = [other for other in questions if other["type"] == question["type"]]
    n_dev = sum(other["split"] == "dev" for other in same_type)
    n_test = sum(other["split"] == "test" for other in same_type)

    return "dev" if n_dev < DEV_SHARE * (n_dev + n_test + 1) else "test"


def main():
    with open(QUESTIONS, encoding="utf-8") as file:
        questions = [json.loads(line) for line in file]

    todo = [question for question in questions if question["split"] is None]
    todo.sort(key=lambda question: question["id"])
    random.Random(SEED).shuffle(todo)
    # Questions that share a gold unit go first: the others rebalance the types afterwards.
    todo.sort(key=lambda question: not find_neighbours(question, questions))

    for question in todo:
        question["split"] = pick_split(question, questions)

    with open(QUESTIONS, "w", encoding="utf-8", newline="\n") as file:
        for question in questions:
            file.write(json.dumps(question, ensure_ascii=False) + "\n")

    print(f"{len(todo)} questions assigned")
    for split in ("dev", "test"):
        types = sorted(question["type"] for question in questions if question["split"] == split)
        counts = {kind: types.count(kind) for kind in dict.fromkeys(types)}
        print(f"{split}: {len(types)} {counts}")


if __name__ == "__main__":
    main()
