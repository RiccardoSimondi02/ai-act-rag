"""Send the questions of a split to the API, then read the stage timings from its logs: percentiles and abstentions.

The answers are also saved in the format of eval_generation.py, so that judge_generation.py judges the service.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_act_rag.pipeline import CONTEXT_WORDS, GENERATOR_MODEL, N_CANDIDATES, RERANKER_MODEL

QUESTIONS = ROOT / "eval" / "dataset" / "questions.jsonl"
SERVICE_RESULTS = ROOT / "eval" / "service_results"
GENERATION_RESULTS = ROOT / "eval" / "generation_results"
STAGES = ["retrieval", "rerank", "prompt", "generation", "other", "total"]


def ask(url: str, question: str) -> tuple[dict, float]:
    request = urllib.request.Request(
        f"{url}/ask", data=json.dumps({"question": question}).encode(), headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
    return body, time.perf_counter() - start


def read_log(log_file: Path, request_ids: set[str]) -> dict[str, dict]:
    """Return the log entries of the given requests, by request id."""

    with open(log_file, encoding="utf-8") as file:
        entries = (json.loads(line) for line in file)
        return {entry["request_id"]: entry for entry in entries if entry.get("request_id") in request_ids}


def generation_row(question: dict, body: dict) -> dict:
    """Return the answer of the service as a row of eval_generation.py: the claims as cleaned by the service."""

    claims = [
        {"text": claim["text"], "sources": [citation["number"] for citation in claim["citations"]]}
        for claim in body["claims"]
    ]
    return {
        "id": question["id"],
        "type": question["type"],
        "question": question["question"],
        "gold_units": question["gold_units"],
        "sources": [source["chunk_id"] for source in body["sources"]],
        "answer": {"abstain": bool(body["abstained"]), "claims": claims} if body["valid_answer"] else None,
        "abstained": body["abstained"],
        "checks": body["checks"],
        "request_id": body["request_id"],
    }


def percentiles(values: list[float]) -> dict[str, float]:
    p50, p95 = np.percentile(values, [50, 95])
    return {"mean": round(float(np.mean(values)), 3), "p50": round(float(p50), 3), "p95": round(float(p95), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs" / "requests.jsonl")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, help="only the first questions, for a quick check")
    args = parser.parse_args()

    with open(QUESTIONS, encoding="utf-8") as file:
        questions = [question for question in map(json.loads, file) if question["split"] == args.split]
    questions = questions[: args.limit]

    sent = []
    answers = []
    for question in questions:
        body, seconds = ask(args.url, question["question"])
        sent.append({"id": question["id"], "type": question["type"], "request_id": body["request_id"], "client_seconds": seconds})
        answers.append(generation_row(question, body))
        print(f"{question['id']} {question['type']:<11} {seconds:6.2f}s abstained={body['abstained']}")

    entries = read_log(args.log_file, {row["request_id"] for row in sent})
    missing = [row["id"] for row in sent if row["request_id"] not in entries]
    if missing:
        sys.exit(f"no log line for {', '.join(missing)}: is --log-file the file the API writes?")
    rows = [row | {"log": entries[row["request_id"]]} for row in sent]
    for row in rows:
        seconds = row["log"]["seconds"]
        # Choice of the context, cleaning of the citations, loading and overhead of Ollama.
        seconds["other"] = seconds["total"] - sum(seconds[stage] for stage in ("retrieval", "rerank", "prompt", "generation"))

    latency = {stage: percentiles([row["log"]["seconds"][stage] for row in rows]) for stage in STAGES}
    latency["client"] = percentiles([row["client_seconds"] for row in rows])

    unanswerable = [row for row in rows if row["type"] == "abstain"]
    answerable = [row for row in rows if row["type"] != "abstain"]
    abstentions = {
        "rate": round(sum(bool(row["log"]["abstained"]) for row in rows) / len(rows), 3),
        "on_abstain_questions": f"{sum(bool(row['log']['abstained']) for row in unanswerable)}/{len(unanswerable)}",
        "on_answerable_questions": f"{sum(bool(row['log']['abstained']) for row in answerable)}/{len(answerable)}",
    }
    scores = {
        "answerable_min": min((row["log"]["top_rerank_score"] for row in answerable), default=None),
        "abstain_max": max((row["log"]["top_rerank_score"] for row in unanswerable), default=None),
    }
    tokens = {kind: percentiles([row["log"]["tokens"][kind] for row in rows]) for kind in ("prompt", "output")}

    print(f"\n{'stage':<12}{'mean':>8}{'p50':>8}{'p95':>8}{'share':>8}")
    for stage, values in latency.items():
        share = values["mean"] / latency["total"]["mean"] if stage in STAGES else None
        print(f"{stage:<12}{values['mean']:>8.3f}{values['p50']:>8.3f}{values['p95']:>8.3f}" + (f"{share:>8.0%}" if share else ""))
    print(f"\nabstentions: {abstentions}\ntop reranker score: {scores}\ntokens: {tokens}")

    out = SERVICE_RESULTS / f"load_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"split": args.split, "questions": len(rows), "latency_seconds": latency, "abstentions": abstentions,
               "top_rerank_score": scores, "tokens": tokens,
               "requests": [{key: row[key] for key in ("id", "type", "request_id", "client_seconds")} for row in rows]}
    with open(out, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    run_name = f"service_{args.split}"
    with open(GENERATION_RESULTS / f"gen_{run_name}.jsonl", "w", encoding="utf-8", newline="\n") as file:
        for answer in answers:
            file.write(json.dumps(answer, ensure_ascii=False) + "\n")
    config = {
        "run": run_name,
        "model": GENERATOR_MODEL,
        "retrieval": {"dense_qwen3_v2_rerank": {"chunking": "v2", "model": RERANKER_MODEL, "n_candidates": N_CANDIDATES}},
        "context_words": CONTEXT_WORDS,
        "split": args.split,
        "source": f"answers of the service at {args.url}",
    }
    with open(GENERATION_RESULTS / f"gen_{run_name}_summary.json", "w", encoding="utf-8") as file:
        json.dump({"config": config}, file, indent=2, ensure_ascii=False)
    print(f"saved {out.relative_to(ROOT)} and {(GENERATION_RESULTS / f'gen_{run_name}.jsonl').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
