"""Check the metrics of a retrieval run against eval/thresholds.json, and exit with 1 if one is below."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
THRESHOLDS = ROOT / "eval" / "thresholds.json"


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def check(summary: dict, thresholds: dict[str, float | None]) -> list[str]:
    """Print each metric next to its threshold, and return the metrics that fail."""

    failed = []
    print(f"{'metric':<14}{'value':>8}{'threshold':>11}")
    for metric, threshold in thresholds.items():
        value = summary.get(metric)
        if value is None:
            status = "missing from the results"
            failed.append(metric)
        elif threshold is None:
            status = "no threshold yet"
        elif value < threshold:
            status = "BELOW"
            failed.append(metric)
        else:
            status = "ok"
        print(f"{metric:<14}{fmt(value):>8}{fmt(threshold):>11}  {status}")
    return failed


def main() -> None:
    with open(THRESHOLDS, encoding="utf-8") as file:
        thresholds = json.load(file)
    if len(sys.argv) != 2 or sys.argv[1] not in thresholds:
        sys.exit(f"Usage: python scripts/check_thresholds.py RUN, where RUN is one of: {', '.join(thresholds)}")
    run_name = sys.argv[1]

    with open(RESULTS / f"{run_name}.json", encoding="utf-8") as file:
        results = json.load(file)
    config, total = results["config"], results["summary"]["total"]
    print(f"{run_name}: split {config['split']}, {total['questions']} questions, device {config.get('device', '-')}")

    failed = check(total, thresholds[run_name])
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("passed")


if __name__ == "__main__":
    main()
