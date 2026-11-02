"""Compare the verdicts of the LLM judge with the labels in eval/labels, criterion by criterion."""

import csv
import json
import math
import warnings

from sklearn.metrics import cohen_kappa_score, confusion_matrix

from eval_generation import RESULTS
from export_labels import LABELS, RAG_RUN

# The values of each criterion, in the order of the rows and columns of the confusion matrix.
VALUES = {
    "correctness": ["correct", "partial", "wrong"],
    "abstention_ok": ["yes", "no"],
    "supported": ["yes", "no"],
}


def yes_no(value: bool | None) -> str | None:
    return None if value is None else ("yes" if value else "no")


def read_labels(name: str) -> list[dict]:
    with open(LABELS / name, encoding="utf-8-sig") as file:
        return list(csv.DictReader(file, delimiter=";"))


def load_judged(run_name: str) -> dict[str, dict]:
    with open(RESULTS / f"judged_{run_name}.jsonl", encoding="utf-8") as file:
        return {row["id"]: row for row in map(json.loads, file)}


def judge_verdict(criterion: str, row: dict) -> tuple[str | None, str | None]:
    """Return the verdict of the judge on one answer for a criterion, as a label value, with its reason."""

    if criterion == "correctness":
        verdict = row["correctness"] or {}
        return verdict.get("verdict"), verdict.get("reason")
    verdict = row["abstention"] or {}
    return yes_no(verdict.get("abstention_ok")), verdict.get("reason")


def answer_pairs(criterion: str, labels: list[dict], judged: dict[str, dict[str, dict]]) -> list[dict]:
    pairs = []
    for label in labels:
        verdict, reason = judge_verdict(criterion, judged[label["run"]][label["id"]])
        pairs.append({"key": f"{label['run']} {label['id']}", "label": label[criterion] or None,
                      "judge": verdict, "reason": reason})
    return pairs


def claim_pairs(labels: list[dict], judged: dict[str, dict]) -> list[dict]:
    pairs = []
    for label in labels:
        claims = {claim["claim_index"]: claim for claim in judged[label["id"]]["claims"]}
        claim = claims.get(int(label["claim_index"]), {})
        pairs.append({"key": f"{label['id']} claim {label['claim_index']}", "label": label["supported"] or None,
                      "judge": yes_no(claim.get("supported")), "reason": claim.get("reason"), "claim": label["claim"]})
    return pairs


def compare(pairs: list[dict], values: list[str]) -> dict:
    """Return agreement, Cohen's kappa, confusion matrix and disagreements of the pairs both sides judged."""

    # A pair empty on both sides is a criterion that does not apply; empty on one side only is reported.
    pairs = [pair for pair in pairs if pair["label"] or pair["judge"]]
    comparable = [pair for pair in pairs if pair["label"] and pair["judge"]]
    labels = [pair["label"] for pair in comparable]
    verdicts = [pair["judge"] for pair in comparable]

    # Kappa is undefined when both sides always give the same value.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = cohen_kappa_score(labels, verdicts, labels=values)
    return {
        "cases": len(comparable),
        "agreement": round(sum(a == b for a, b in zip(labels, verdicts)) / len(comparable), 3),
        "kappa": None if math.isnan(kappa) else round(float(kappa), 3),
        "confusion_matrix": {"rows_labels_columns_judge": values,
                             "matrix": confusion_matrix(labels, verdicts, labels=values).tolist()},
        "disagreements": [pair for pair in comparable if pair["label"] != pair["judge"]],
        "judged_on_one_side_only": [pair for pair in pairs if not (pair["label"] and pair["judge"])],
    }


def print_result(criterion: str, result: dict) -> None:
    values = result["confusion_matrix"]["rows_labels_columns_judge"]
    print(f"\n{criterion}: {result['cases']} cases, agreement {result['agreement']}, kappa {result['kappa']}")
    print(f"{'label \\ judge':<16}" + "".join(f"{value:>9}" for value in values))
    for value, counts in zip(values, result["confusion_matrix"]["matrix"]):
        print(f"{value:<16}" + "".join(f"{count:>9}" for count in counts))
    for pair in result["disagreements"]:
        print(f"  {pair['key']}: label {pair['label']}, judge {pair['judge']} - {pair['reason']}")
    for pair in result["judged_on_one_side_only"]:
        print(f"  {pair['key']}: only one side judged (label {pair['label']}, judge {pair['judge']})")


def main() -> None:
    answer_labels = read_labels("answers.csv")
    judged = {run_name: load_judged(run_name) for run_name in sorted({label["run"] for label in answer_labels})}

    results = {criterion: compare(answer_pairs(criterion, answer_labels, judged), VALUES[criterion])
               for criterion in ("correctness", "abstention_ok")}
    results["supported"] = compare(claim_pairs(read_labels("claims.csv"), judged[RAG_RUN]), VALUES["supported"])

    for criterion, result in results.items():
        print_result(criterion, result)
    with open(RESULTS / "judge_agreement.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"\nsaved {(RESULTS / 'judge_agreement.json').name}")


if __name__ == "__main__":
    main()
