from ai_act_rag.generation.verify import cited_units, cites_gold, coverage, valid_sources

# Sources [1], [2], [3] given to the model, with the units each chunk comes from.
SOURCE_UNIT_IDS = [["art_99.par_3"], ["art_5.par_1"], ["art_5.par_1"]]
ANSWER = {
    "abstain": False,
    "claims": [
        {"text": "Fines go up to EUR 35 000 000.", "sources": [1]},
        {"text": "The practices are listed in Article 5.", "sources": [3, 2]},
    ],
}
ABSTENTION = {"abstain": True, "claims": []}


def test_valid_sources_rejects_numbers_outside_the_context():
    assert valid_sources(ANSWER, n_sources=3)
    assert not valid_sources(ANSWER, n_sources=2)
    assert not valid_sources({"abstain": False, "claims": [{"text": "x", "sources": [0]}]}, n_sources=3)


def test_coverage_needs_a_source_for_every_claim():
    assert coverage(ANSWER)
    assert not coverage({"abstain": False, "claims": [*ANSWER["claims"], {"text": "Uncited.", "sources": []}]})


def test_cited_units_translate_source_numbers_into_units():
    assert cited_units(ANSWER, SOURCE_UNIT_IDS) == ["art_99.par_3", "art_5.par_1"]
    # A number outside the context is skipped, not an error: valid_sources reports it.
    assert cited_units({"abstain": False, "claims": [{"text": "x", "sources": [7, 1]}]}, SOURCE_UNIT_IDS) == [
        "art_99.par_3"
    ]


def test_cites_gold():
    assert cites_gold(ANSWER, SOURCE_UNIT_IDS, ["art_99.par_3"]) is True
    assert cites_gold(ANSWER, SOURCE_UNIT_IDS, ["art_50.par_1"]) is False
    assert cites_gold(ABSTENTION, SOURCE_UNIT_IDS, ["art_99.par_3"]) is False
    assert cites_gold(ABSTENTION, SOURCE_UNIT_IDS, []) is None


def test_an_abstention_passes_the_first_two_checks():
    assert valid_sources(ABSTENTION, n_sources=3)
    assert coverage(ABSTENTION)
