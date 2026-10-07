import json
from collections import Counter
from pathlib import Path

import pytest

from ai_act_rag.ingestion.parse_eurlex import load_soup, parse_units

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ai_act.html"
QUESTIONS = ROOT / "eval" / "dataset" / "questions.jsonl"


@pytest.fixture(scope="session")
def questions():
    with QUESTIONS.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


# data/processed is not versioned: the ids come from the raw file, not from units.jsonl.
@pytest.fixture(scope="session")
def unit_ids():
    return {unit["id"] for unit in parse_units(load_soup(RAW))}


def test_question_ids_are_unique(questions):
    counts = Counter(question["id"] for question in questions)
    assert [question_id for question_id, count in counts.items() if count > 1] == []


def test_every_gold_unit_exists(questions, unit_ids):
    missing = [
        (question["id"], unit_id)
        for question in questions
        for unit_id in question["gold_units"]
        if unit_id not in unit_ids
    ]
    assert missing == []


def test_only_abstain_questions_have_no_gold_units(questions):
    for question in questions:
        expects_abstention = question["type"] == "abstain"
        assert (question["gold_units"] == []) == expects_abstention, question["id"]
