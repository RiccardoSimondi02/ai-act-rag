import json
from pathlib import Path

import pytest

from ai_act_rag.ingestion.parse_eurlex import (
    ANNEX_MAX_WORDS,
    EXCLUDED_ARTICLES,
    coverage,
    find_annexes,
    find_articles,
    load_soup,
    parse_article_header,
    parse_units,
    write_jsonl,
)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "ai_act.html"
BASE_ACT = "32024R1689"
AMENDING_ACT = "32026R1744"


@pytest.fixture(scope="session")
def soup():
    return load_soup(RAW)


@pytest.fixture(scope="session")
def units(soup):
    return {unit["id"]: unit for unit in parse_units(soup)}


# --- load_soup ---------------------------------------------------------------


def test_load_soup_reads_title_as_utf8(soup):
    # With the wrong encoding the dashes would become "â€”".
    assert soup.title.string == "Consolidated TEXT: 32024R1689 — EN — 27.07.2026"


# --- find_articles -----------------------------------------------------------


def test_find_articles(soup):
    assert len(find_articles(soup)) == 119


def test_find_annexes(soup):
    assert len(find_annexes(soup)) == 14


# --- parse_article_header ----------------------------------------------------


def test_parse_article_header(soup):
    headers = [parse_article_header(article) for article in find_articles(soup)[:5]]
    assert [(h["parent_number"], h["parent_title"]) for h in headers] == [
        ("1", "Subject matter"),
        ("2", "Scope"),
        ("3", "Definitions"),
        ("4", "AI literacy"),
        (
            "4a",
            "Processing of special categories of personal data for bias detection and correction",
        ),
    ]


# --- parse_units -------------------------------------------------------------


def test_units_are_unique_and_clean(soup, units):
    assert len(units) == len(parse_units(soup)) == 673
    for unit in units.values():
        assert unit["text"]
        assert unit["act_at_start"] in {BASE_ACT, AMENDING_ACT}
        assert unit["source_url"].endswith("/eng#" + unit["parent_id"])
        assert "▼" not in unit["text"]
        assert "\xa0" not in unit["text"]


@pytest.mark.parametrize(
    ("unit_id", "act"),
    [
        ("art_99.par_1", AMENDING_ACT),
        ("art_99.par_3", BASE_ACT),
        ("art_1.par_1", BASE_ACT),
        ("art_4a.par_1", AMENDING_ACT),
        ("art_57.par_10", AMENDING_ACT),
        ("art_58.par_2", BASE_ACT),
        ("art_3.pt_14a", AMENDING_ACT),
        ("art_3.pt_15", BASE_ACT),
        ("art_75b", AMENDING_ACT),
        ("anx_III.pt_6", BASE_ACT),
        ("anx_XIV", AMENDING_ACT),
    ],
)
def test_act_at_start(units, unit_id, act):
    assert units[unit_id]["act_at_start"] == act


def test_text_of_article_99_paragraph_3(units):
    assert units["art_99.par_3"]["text"] == (
        "Non-compliance with the prohibition of the AI practices referred to in Article 5 "
        "shall be subject to administrative fines of up to EUR 35 000 000 or, if the "
        "offender is an undertaking, up to 7 % of its total worldwide annual turnover for "
        "the preceding financial year, whichever is higher."
    )


def test_article_3_has_one_unit_per_definition(units):
    definitions = [unit for unit in units.values() if unit["parent_id"] == "art_3"]
    assert len(definitions) == 70
    assert units["art_3.pt_12"]["citation"] == "Article 3(12)"
    assert units["art_3.pt_12"]["point_number"] == "12"
    assert units["art_3.pt_12"]["paragraph_number"] is None
    assert units["art_3.pt_1"]["text"].startswith("‘AI system’ means a machine-based system")


def test_article_without_paragraphs_is_one_unit(units):
    unit = units["art_16"]
    assert unit["citation"] == "Article 16"
    assert unit["paragraph_number"] is None
    assert unit["point_number"] is None
    assert unit["text"].startswith("Providers of high-risk AI systems shall: (a) ensure")
    assert unit["text"].endswith("Directives (EU) 2016/2102 and (EU) 2019/882.")


def test_short_annexes_are_one_unit_and_long_ones_are_split(units):
    per_annex = {}
    for unit in units.values():
        if unit["parent_id"].startswith("anx_"):
            per_annex.setdefault(unit["parent_id"], []).append(unit)

    assert len(per_annex) == 14
    assert {annex_id: len(parts) for annex_id, parts in per_annex.items() if len(parts) > 1} == {
        "anx_I": 2,
        "anx_III": 8,
        "anx_IV": 9,
        "anx_VII": 5,
        "anx_VIII": 3,
        "anx_X": 7,
    }
    for annex_id, parts in per_annex.items():
        words = sum(len(unit["text"].split()) for unit in parts)
        assert (len(parts) == 1) == (words <= ANNEX_MAX_WORDS), annex_id

    assert units["anx_II"]["citation"] == "Annex II"
    assert units["anx_III.pt_6"]["citation"] == "Annex III, point 6"
    assert units["anx_III.pt_6"]["text"].startswith("6. Law enforcement")
    assert units["anx_I.sec_A"]["citation"] == "Annex I, Section A"
    assert units["anx_III.pt_6"]["parent_title"] == "High-risk AI systems referred to in Article 6(2)"


def test_introduction_is_on_every_unit_it_introduces(units):
    intro = "High-risk AI systems pursuant to Article 6(2) are the AI systems listed in any of the following areas:"
    assert units["anx_III.pt_1"]["parent_intro"] == units["anx_III.pt_8"]["parent_intro"] == intro
    assert units["art_3.pt_12"]["parent_intro"] == (
        "For the purposes of this Regulation, the following definitions apply:"
    )
    assert units["anx_IV.pt_5"]["parent_intro"].startswith("The technical documentation")
    assert units["art_99.par_3"]["parent_intro"] is None
    assert units["anx_II"]["parent_intro"] is None


def test_every_word_of_the_source_is_in_a_unit(soup, units):
    shares = coverage(soup, list(units.values()))
    excluded = {f"art_{number}" for number in EXCLUDED_ARTICLES}
    assert len(shares) == 119 + 14
    for parent_id, share in shares.items():
        assert share == (0 if parent_id in excluded else 1), parent_id


def test_unnumbered_subparagraphs_join_their_paragraph(units):
    text = units["art_6.par_3"]["text"]
    assert "The first subparagraph shall apply where" in text
    assert text.endswith("where the AI system performs profiling of natural persons.")


def test_exponent_in_article_51_paragraph_2(units):
    assert units["art_51.par_2"]["text"].endswith("is greater than 10^25.")


# --- write_jsonl -------------------------------------------------------------


def test_write_jsonl_round_trip(tmp_path):
    records = [{"id": "a", "text": "person’s — 7 %"}, {"id": "b", "text": "second"}]
    path = tmp_path / "processed" / "units.jsonl"

    write_jsonl(records, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records
    assert "’" in lines[0]
