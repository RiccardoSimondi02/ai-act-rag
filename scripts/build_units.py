"""Build data/processed/units.jsonl from the consolidated HTML of the AI Act."""

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_act_rag.ingestion.parse_eurlex import ( 
    coverage,
    load_soup,
    parse_units,
    write_jsonl,
)

RAW = ROOT / "data" / "raw" / "ai_act.html"
OUT = ROOT / "data" / "processed" / "units.jsonl"


def main() -> None:
    soup = load_soup(RAW)
    units = parse_units(soup)
    write_jsonl(units, OUT)

    shares = coverage(soup, units)
    partial = [f"{parent_id} {share:.1%}" for parent_id, share in shares.items() if share != 1]

    print(f"{len(units)} records written to {OUT.relative_to(ROOT)}")
    print(f"by act at start: {dict(Counter(unit['act_at_start'] for unit in units))}")
    print(f"articles and annexes not fully covered: {len(partial)} of {len(shares)} -> {', '.join(partial)}")


if __name__ == "__main__":
    main()
