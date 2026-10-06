import copy
import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, Tag

# Ids like "art_99.tit_1" belong to div.eli-title: the class filters them out.
ARTICLE = 'div.eli-subdivision[id^="art_"]'
# Direct children of the article only
PARAGRAPH = f"{ARTICLE} > div.norm:has(> span.no-parag)"
DEFINITION = "#art_3 > div.grid-list"
# "For the purposes of this Regulation, the following definitions apply:"
DEFINITION_INTRO = "#art_3 > p.norm"
# First element after the title, in articles that have no numbered paragraphs.
WHOLE_ARTICLE = f"{ARTICLE}:not(#art_3, :has(> div.norm > span.no-parag)) > div.eli-title + p.norm"
# Unnumbered subparagraphs and lists: siblings of the numbered paragraph they belong to.
CONTINUATION = f"{ARTICLE}:has(> div.norm > span.no-parag) > :is(p.norm, div.grid-list)"

ANNEX = 'div[id^="anx_"]'
# "ANNEX III": the first title of each annex. Annex X uses the same class for its second title.
ANNEX_START = f"{ANNEX} > p.title-annex-1:not(p.title-annex-1 ~ p.title-annex-1)"
ANNEX_CHILD = f"{ANNEX} > :not(hr, p.modref, p.title-annex-1, p.title-annex-2)"
# Where a unit starts in an annex that is split: its section headings if it has any,
# its numbered points otherwise.
ANNEX_BLOCK = (
    f"{ANNEX}:has(> p.title-gr-seq-level-1) > p.title-gr-seq-level-1, "
    f"{ANNEX}:not(:has(> p.title-gr-seq-level-1)) > div.grid-list"
)

# The first ▼B of the document is a p.arrow, every other marker is a p.modref.
MARKER = "p.modref, p.arrow"
NOT_TEXT = "p.modref, p.title-article-norm, div.eli-title, p.title-annex-1, p.title-annex-2"

# Articles 102-110 only amend other acts.
EXCLUDED_ARTICLES = {str(number) for number in range(102, 111)}
# Annexes up to this size stay whole, like the articles without numbered paragraphs.
ANNEX_MAX_WORDS = 500

# Typos in the source
TITLE_FIXES = {"Subject matter'": "Subject matter"}

# "Section A. List of…" -> ("Section ", "A"), "3. Quality management system" -> (None, "3")
_BLOCK_LABEL_RE = re.compile(r"(Section )?(\w+)")


def load_soup(path: str | Path) -> BeautifulSoup:
    """Read the HTML file and return the BeautifulSoup document."""

    text = Path(path).read_text(encoding="utf-8")
    return BeautifulSoup(text, "lxml")


def find_articles(soup: BeautifulSoup) -> list[Tag]:
    """Return the article divs, in document order."""

    return soup.select(ARTICLE)


def find_annexes(soup: BeautifulSoup) -> list[Tag]:
    """Return the annex divs, in document order."""

    return soup.select(ANNEX)


def clean_text(text: str) -> str:
    """Normalise the whitespace left by get_text(" ")."""

    text = " ".join(text.split())
    text = re.sub(r" ([,;:.)])", r"\1", text)
    return text.replace("( ", "(")


def extract_text(element: Tag) -> str:
    """Return the text of an element, without titles and amendment markers."""

    # Works on a copy, so parse_units still meets the nested markers during its scan.
    element = copy.copy(element)
    for dropped in element.select(NOT_TEXT):
        dropped.decompose()

    # Exponents: 10<span class="superscript">25</span> would read "10 25", so it becomes "10^25".
    # Footnote references are superscripts inside an <a> and are left alone.
    for exponent in element.select("span.superscript"):
        if exponent.find_parent("a") is None:
            exponent.replace_with("^" + exponent.get_text(strip=True))
    element.smooth()
    return clean_text(element.get_text(" ", strip=True))


def parse_article_header(article_div: Tag) -> dict:
    """Return the id, number and title of an article."""

    label = article_div.select_one("p.title-article-norm").get_text(" ", strip=True)
    title = clean_text(article_div.select_one("p.stitle-article-norm").get_text(" ", strip=True))
    return {
        "parent_id": article_div["id"],
        # "Article\xa04a" -> "4a": split() handles both the regular space and \xa0.
        "parent_number": label.split()[-1],
        "parent_title": TITLE_FIXES.get(title, title),
    }


def parse_annex_header(annex_div: Tag) -> dict:
    """Return the id, number and title of an annex."""

    label, *titles = annex_div.select(":scope > :is(p.title-annex-1, p.title-annex-2)")
    return {
        "parent_id": annex_div["id"],
        "parent_number": label.get_text(" ", strip=True).split()[-1],
        # Annex XIV has no title element.
        "parent_title": clean_text(titles[0].get_text(" ", strip=True)) if titles else None,
    }


def parse_paragraph(paragraph_div: Tag) -> dict:
    """Return the number and text of a paragraph."""

    number = paragraph_div.find("span", class_="no-parag", recursive=False)
    body = paragraph_div.find("div", class_="inline-element", recursive=False)
    return {
        "paragraph_number": number.get_text(strip=True).rstrip("."),
        "text": extract_text(body),
    }


def parse_definition(definition_div: Tag) -> dict:
    """Return the number and text of a definition of Article 3."""

    number = definition_div.select_one(":scope > div.grid-list-column-1")
    body = definition_div.select_one(":scope > div.grid-list-column-2")
    return {
        "point_number": number.get_text(strip=True).strip("()"),
        "text": extract_text(body),
    }


def unexpected(element: Tag) -> ValueError:
    """Build the error for an element that parse_units does not know where to put."""

    text = element.get_text(" ", strip=True)[:80]
    return ValueError(f"Unexpected <{element.name}> in {element.parent['id']}: {text!r}")


def parse_units(soup: BeautifulSoup) -> list[dict]:
    """Return one record per unit, with the CELEX of the act in force where the unit starts.

    In the articles a unit is a numbered paragraph with its unnumbered subparagraphs, a
    definition of Article 3, or a whole article when the article has no numbered paragraphs.

    In the annexes a unit is the whole annex up to ANNEX_MAX_WORDS, otherwise one of its
    sections or numbered points. Annex units keep their label ("Section A.", "3.") in the text.

    When an article or annex is split into points, the sentence that introduces them is
    not a unit: it goes in the parent_intro field of each of them.

    A marker applies until the next one, even across article boundaries, and it can
    sit outside the articles or inside a paragraph.

    Raises ValueError on a structure it does not know, instead of dropping the text.
    """

    canonical = soup.select_one('link[rel="canonical"]')
    source_url = canonical["href"] if canonical else None

    units = []
    intros = {}
    act = None
    annex = None

    def started(parent: dict) -> bool:
        return bool(units) and units[-1]["parent_id"] == parent["parent_id"]

    def set_intro(parent: dict, element: Tag) -> None:
        if started(parent) or parent["parent_id"] in intros:
            raise unexpected(element)
        intros[parent["parent_id"]] = extract_text(element)

    def add_unit(parent: dict, text: str, id_suffix: str = "", cited: str = "", **numbers) -> None:
        kind = "Annex" if parent["parent_id"].startswith("anx_") else "Article"
        units.append(
            {
                # "art_99.par_3", "art_3.pt_12", "art_16", "anx_III.pt_6", "anx_I.sec_A", "anx_II"
                "id": parent["parent_id"] + id_suffix,
                **parent,
                "parent_intro": intros.get(parent["parent_id"]),
                "paragraph_number": numbers.get("paragraph_number"),
                "point_number": numbers.get("point_number"),
                "text": text,
                # "Article 99(3)", "Article 16", "Annex III, point 6", "Annex I, Section A"
                "citation": f"{kind} {parent['parent_number']}{cited}",
                # Later markers inside the unit are not recorded: the text may be amended further on.
                "act_at_start": act,
                "source_url": f"{source_url}#{parent['parent_id']}" if source_url else None,
            }
        )

    selectors = (
        f"{MARKER}, {PARAGRAPH}, {CONTINUATION}, {DEFINITION}, {DEFINITION_INTRO}, "
        f"{WHOLE_ARTICLE}, {ANNEX_START}, {ANNEX_CHILD}"
    )
    for element in soup.select(selectors):
        if element.css.match(MARKER):
            # The legend at the top uses "►": only "▼" opens a block of text.
            if "▼" in element.get_text():
                # The link title is "32024R1689" or "32026R1744: REPLACED".
                act = element.a["title"].split(":")[0]
            continue

        if element.css.match(ANNEX_START):
            annex = parse_annex_header(element.parent)
            text = extract_text(element.parent)
            if len(text.split()) <= ANNEX_MAX_WORDS:
                add_unit(annex, text)
                annex = None
            continue

        if element.css.match(ANNEX_CHILD):
            if annex is None:
                continue
            text = extract_text(element)
            if element.css.match(ANNEX_BLOCK):
                match = _BLOCK_LABEL_RE.match(text)
                if not match:
                    raise unexpected(element)
                section, label = match.groups()
                if section:
                    add_unit(annex, text, f".sec_{label}", f", Section {label}")
                else:
                    add_unit(annex, text, f".pt_{label}", f", point {label}", point_number=label)
            elif not text:
                # Empty <p> used as spacing.
                continue
            elif started(annex):
                units[-1]["text"] += " " + text
            else:
                set_intro(annex, element)
            continue

        article = parse_article_header(element.parent)
        if article["parent_number"] in EXCLUDED_ARTICLES:
            continue

        if element.css.match(DEFINITION_INTRO):
            set_intro(article, element)
        elif element.css.match(CONTINUATION):
            if not started(article):
                raise unexpected(element)
            units[-1]["text"] += " " + extract_text(element)
        elif element.css.match(PARAGRAPH):
            paragraph = parse_paragraph(element)
            number = paragraph["paragraph_number"]
            add_unit(article, paragraph["text"], f".par_{number}", f"({number})", paragraph_number=number)
        elif element.css.match(DEFINITION):
            definition = parse_definition(element)
            number = definition["point_number"]
            add_unit(article, definition["text"], f".pt_{number}", f"({number})", point_number=number)
        else:
            add_unit(article, extract_text(element.parent))
    return units


def coverage(soup: BeautifulSoup, units: list[dict]) -> dict[str, float]:
    """Return, for each article and annex, the share of its words that ended up in the units."""

    words = Counter()
    for unit in units:
        # Article units have their number stripped from the text, annex units keep it.
        numbered = bool(unit["paragraph_number"] or unit["point_number"])
        stripped = numbered and unit["parent_id"].startswith("art_")
        words[unit["parent_id"]] += len(unit["text"].split()) + stripped
    # An introduction is repeated on every unit of its parent: it counts once.
    for parent_id, intro in {(unit["parent_id"], unit["parent_intro"]) for unit in units}:
        words[parent_id] += len(intro.split()) if intro else 0
    return {
        parent["id"]: words[parent["id"]] / len(extract_text(parent).split())
        for parent in soup.select(f"{ARTICLE}, {ANNEX}")
    }


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write one JSON record per line, creating the folder if it is missing."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
