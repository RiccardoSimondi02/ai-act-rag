from pathlib import Path
import re
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "data" / "processed" / "units.jsonl"


def main():

    if len(sys.argv) < 2:
        sys.exit("insert word")
    word = sys.argv[1]

    count = 0
    with open(UNITS, encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            for field in ("text", "parent_intro", "parent_title"):
                match = re.search(re.escape(word), record[field] or "", re.IGNORECASE)
                if match:
                    count += 1
                    start = max(0, match.start() - 60) 
                    excerpt =  record[field][start: match.end() + 60]
                    print(record["id"], record["parent_title"] or "", record["citation"], excerpt)
                    break
    print(f"{count} units")


if __name__ == "__main__":
    main()
