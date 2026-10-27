"""Print the questions whose gold units changed position between two runs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"


def load_run(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        sys.exit(f"no results for {name}: run eval_retrieval.py {name} first")
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def gold_positions(run: dict) -> dict:
    return {row["id"]: row["gold_positions"] for row in run["questions"]}


def rank(position: int | None) -> float:
    """Return the position as a number, with a gold unit that was not found after all the others."""

    return float("inf") if position is None else position


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Usage: python scripts/compare_runs.py BEFORE AFTER")
    names = sys.argv[1:]
    before, after = (load_run(name) for name in names)

    old, new = gold_positions(before), gold_positions(after)
    if {key: sorted(value) for key, value in old.items()} != {key: sorted(value) for key, value in new.items()}:
        sys.exit("the two runs do not cover the same questions and gold units")

    types = {row["id"]: row["type"] for row in before["questions"]}
    depth = before["config"]["depth"]
    up = down = 0
    for question_id in old:
        for unit_id, position in old[question_id].items():
            moved_to = new[question_id][unit_id]
            if moved_to == position:
                continue
            better = rank(moved_to) < rank(position)
            up += better
            down += not better
            shown = [f">{depth}" if value is None else str(value) for value in (position, moved_to)]
            print(
                f"{question_id:<6}{types[question_id]:<12}{unit_id:<16}"
                f"{shown[0]:>5} -> {shown[1]:<6}{'better' if better else 'worse'}"
            )

    print(f"{up} gold units moved up, {down} moved down")
    columns = ["recall@5", "recall@10", "recall@20", "recall@1000w", "recall@2000w", "mrr"]
    print(f"{'run':<22}" + "".join(f"{column:>14}" for column in columns))
    for name, run in zip(names, (before, after)):
        total = run["summary"]["total"]
        print(f"{name:<22}" + "".join(f"{total[column]:>14.3f}" if column in total else f"{'-':>14}" for column in columns))


if __name__ == "__main__":
    main()
