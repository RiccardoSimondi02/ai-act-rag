"""Checks of the citations in an answer.

An answer is {"abstain": bool, "claims": [{"text": str, "sources": [int]}]}, where each
source number refers to the numbered sources given to the model, starting from 1.
"""


def valid_sources(answer: dict, n_sources: int) -> bool:
    """Return whether every cited number is one of the sources given to the model."""

    return all(1 <= number <= n_sources for claim in answer["claims"] for number in claim["sources"])


def coverage(answer: dict) -> bool:
    """Return whether every claim cites at least one source."""

    return all(claim["sources"] for claim in answer["claims"])


def cited_units(answer: dict, source_unit_ids: list[list[str]]) -> list[str]:
    """Return the unit ids of the cited sources, in order and without repeats; invalid numbers are skipped."""

    units = []
    for claim in answer["claims"]:
        for number in claim["sources"]:
            if 1 <= number <= len(source_unit_ids):
                units.extend(unit_id for unit_id in source_unit_ids[number - 1] if unit_id not in units)
    return units


def cites_gold(answer: dict, source_unit_ids: list[list[str]], gold: list[str]) -> bool | None:
    """Return whether the answer cites at least one gold unit, or None when the question has no gold."""

    if not gold:
        return None
    return bool(set(cited_units(answer, source_unit_ids)) & set(gold))
