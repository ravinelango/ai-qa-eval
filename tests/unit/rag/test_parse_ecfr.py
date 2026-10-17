"""eCFR parsing on a synthetic part: labels, nesting, headings and loud failures."""

from __future__ import annotations

import pytest

from apps.rag.errors import SourceParseError
from apps.rag.parse_ecfr import (
    DIGIT,
    LETTER,
    ROMAN,
    UPPER,
    assign_level,
    leading_markers,
    ordinal,
    parse_ecfr_part,
)
from apps.rag.units import SourceUnit
from tests.rag_helpers import SYNTHETIC_ECFR


@pytest.fixture(scope="module")
def units() -> dict[str, SourceUnit]:
    parsed = parse_ecfr_part(SYNTHETIC_ECFR.read_bytes(), doc_id="synthetic-part99", part="99")
    by_locator = {u.locator: u for u in parsed}
    assert len(by_locator) == len(parsed), "locators must be unique"
    return by_locator


def test_units_follow_document_order_and_paragraph_level(units: dict[str, SourceUnit]) -> None:
    assert list(units) == [
        "99.1(a)",
        "99.1(b)",
        "99.3(a)",
        "99.3(b)",
        "99.10-intro",
        *[f"99.10({c})" for c in "abcdefghijk"],
        "99.30",
        "99.50(a)",
        "99.50(b)",
        "99.200(a)",
        "99.200(b)",
        "99.200(c)",
        *[f"99.300({c})" for c in "abcde"],
    ]


def test_labels_and_heading_paths(units: dict[str, SourceUnit]) -> None:
    unit = units["99.10(e)"]
    assert unit.label == "§99.10(e)"
    assert unit.heading_path == (
        "PART 99—SYNTHETIC RECORDS; SYNTHETIC SIGNATURES",
        "Subpart B—Synthetic Ledgers",
        "§ 99.10 Controls for sealed ledgers.",
    )
    assert unit.blocks[0].text.startswith("(e) Use of placeholder change logs")
    assert unit.blocks[0].element == "DIV8[§ 99.10]/P[6]"


def test_letter_i_after_h_is_a_paragraph_not_a_roman_numeral(
    units: dict[str, SourceUnit],
) -> None:
    assert units["99.10(i)"].blocks[0].paragraph == "§99.10(i)"


def test_subparagraphs_stay_with_their_top_level_paragraph(units: dict[str, SourceUnit]) -> None:
    labels = [b.paragraph for b in units["99.50(a)"].blocks]
    assert labels == ["§99.50(a)", "§99.50(a)(1)", "§99.50(a)(2)", "§99.50(a)(3)"]
    roman = [b.paragraph for b in units["99.200(a)"].blocks]
    assert roman == [
        "§99.200(a)",
        "§99.200(a)(1)",
        "§99.200(a)(1)(i)",
        "§99.200(a)(1)(ii)",
        "§99.200(a)(2)",
    ]


def test_paragraph_opening_with_two_markers_starts_a_new_unit(
    units: dict[str, SourceUnit],
) -> None:
    blocks = units["99.200(b)"].blocks
    assert [b.paragraph for b in blocks] == ["§99.200(b)(1)", "§99.200(b)(2)", "§99.200(b)(2)"]
    assert blocks[2].text == "Placeholder flush text that continues paragraph (b)(2)."


def test_intro_text_and_unlabeled_sections(units: dict[str, SourceUnit]) -> None:
    intro = units["99.10-intro"]
    assert intro.label == "§99.10 (introductory text)"
    assert intro.blocks[0].paragraph is None
    single = units["99.30"]
    assert single.label == "§99.30"
    assert len(single.blocks) == 1


def test_italic_terms_are_kept_and_citations_dropped(units: dict[str, SourceUnit]) -> None:
    assert units["99.3(b)"].blocks[1].text == "(1) Widget means a placeholder object."
    text = " ".join(b.text for u in units.values() for b in u.blocks)
    assert "Synthetic citation" not in text
    assert "Synthetic authority" not in text


@pytest.mark.parametrize(
    ("marker", "stack", "expected"),
    [
        ("a", [], [(LETTER, "a")]),
        ("i", [(LETTER, "h")], [(LETTER, "i")]),
        ("i", [(LETTER, "a"), (DIGIT, "1")], [(LETTER, "a"), (DIGIT, "1"), (ROMAN, "i")]),
        (
            "v",
            [(LETTER, "a"), (DIGIT, "1"), (ROMAN, "iv")],
            [(LETTER, "a"), (DIGIT, "1"), (ROMAN, "v")],
        ),
        (
            "x",
            [(LETTER, "a"), (DIGIT, "2"), (ROMAN, "ix")],
            [(LETTER, "a"), (DIGIT, "2"), (ROMAN, "x")],
        ),
        ("c", [(LETTER, "b"), (DIGIT, "1"), (ROMAN, "ii")], [(LETTER, "c")]),
        (
            "A",
            [(LETTER, "a"), (DIGIT, "1"), (ROMAN, "i")],
            [(LETTER, "a"), (DIGIT, "1"), (ROMAN, "i"), (UPPER, "A")],
        ),
        ("aa", [(LETTER, "z")], [(LETTER, "aa")]),
    ],
)
def test_assign_level(
    marker: str, stack: list[tuple[int, str]], expected: list[tuple[int, str]]
) -> None:
    assert assign_level(marker, stack, "test") == expected


@pytest.mark.parametrize(
    ("marker", "stack"),
    [("c", [(LETTER, "a")]), ("2", []), ("iii", [(LETTER, "a"), (DIGIT, "1")])],
)
def test_out_of_sequence_markers_stop_the_parse(marker: str, stack: list[tuple[int, str]]) -> None:
    with pytest.raises(SourceParseError, match="does not follow"):
        assign_level(marker, stack, "test")


@pytest.mark.parametrize(
    ("following", "expected"),
    [
        ("ii", [(LETTER, "h"), (DIGIT, "1"), (ROMAN, "i")]),
        ("2", [(LETTER, "h"), (DIGIT, "1"), (ROMAN, "i")]),
        ("j", [(LETTER, "i")]),
    ],
)
def test_ambiguous_marker_is_resolved_by_the_next_one(
    following: str, expected: list[tuple[int, str]]
) -> None:
    # "(i)" after "(h)(1)" is either paragraph (i) or (h)(1)(i).
    assert assign_level("i", [(LETTER, "h"), (DIGIT, "1")], "test", following) == expected


def test_unresolvable_ambiguity_stops_the_parse() -> None:
    with pytest.raises(SourceParseError, match="ambiguous"):
        assign_level("i", [(LETTER, "h"), (DIGIT, "1")], "test", None)


def test_ordinals() -> None:
    assert ordinal(LETTER, "i") == 9
    assert ordinal(ROMAN, "i") == 1
    assert ordinal(ROMAN, "xiv") == 14
    assert ordinal(ROMAN, "iiii") is None
    assert ordinal(LETTER, "ab") is None
    assert leading_markers("(b)(1) Text (c) inside") == (["b", "1"], "Text (c) inside")


def _xml(section_body: str) -> bytes:
    return (
        '<DIV5 N="99" TYPE="PART"><HEAD>PART 99—T</HEAD>'
        f'<DIV8 N="§ 99.1" TYPE="SECTION"><HEAD>§ 99.1 Scope.</HEAD>{section_body}</DIV8>'
        "</DIV5>"
    ).encode()


def test_element_locations_count_each_tag_separately() -> None:
    units = parse_ecfr_part(
        _xml("<P>(a) one</P><FP>flush</FP><P>(b) two</P>"), doc_id="d", part="99"
    )
    elements = [b.element for u in units for b in u.blocks]
    assert elements == ["DIV8[§ 99.1]/P[1]", "DIV8[§ 99.1]/FP[1]", "DIV8[§ 99.1]/P[2]"]


def test_unknown_elements_stop_the_parse_instead_of_being_dropped() -> None:
    with pytest.raises(SourceParseError, match="unexpected element <EXTRACT>"):
        parse_ecfr_part(_xml("<P>(a) x</P><EXTRACT>y</EXTRACT>"), doc_id="d", part="99")


def test_missing_part_or_section_number_stops_the_parse() -> None:
    with pytest.raises(SourceParseError, match="exactly one PART 11"):
        parse_ecfr_part(_xml("<P>(a) x</P>"), doc_id="d", part="11")
    bad_head = (
        b'<DIV5 N="99" TYPE="PART"><HEAD>P</HEAD>'
        b'<DIV8 TYPE="SECTION"><HEAD>Scope</HEAD><P>x</P></DIV8></DIV5>'
    )
    with pytest.raises(SourceParseError, match="section number"):
        parse_ecfr_part(bad_head, doc_id="d", part="99")
    with pytest.raises(SourceParseError, match="not well-formed"):
        parse_ecfr_part(b"<DIV5", doc_id="d", part="99")
