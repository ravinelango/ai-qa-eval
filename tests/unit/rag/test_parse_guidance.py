"""Guidance PDF parsing: headings from the table of contents, page furniture, running
line numbers, paragraphs and page tracking. All text here is synthetic."""

from __future__ import annotations

import pytest

from apps.rag.corpus import PdfOptions
from apps.rag.errors import SourceParseError
from apps.rag.parse_guidance import PdfText, parse_guidance, read_pdf, toc_headings
from apps.rag.units import SourceUnit
from tests.rag_helpers import GUIDANCE_PAGES, make_pdf

TITLE = "Synthetic Guidance on Ledgers (test fixture)"
TOC = PdfOptions(toc_pages=(1,))


def _parse(pdf: PdfText, options: PdfOptions = TOC) -> dict[str, SourceUnit]:
    units = parse_guidance(pdf, doc_id="g", title=TITLE, options=options)
    return {u.locator: u for u in units}


def _toc(*entries: str) -> str:
    return "TABLE OF CONTENTS\n" + "\n".join(f"{e} ........ 1" for e in entries)


def test_read_pdf_and_sections_from_the_table_of_contents() -> None:
    pdf = read_pdf(make_pdf(GUIDANCE_PAGES))
    assert len(pdf.pages) == 4
    units = _parse(pdf, PdfOptions(toc_pages=(2,)))
    # "II. DISCUSSION" has no text of its own before "A.", so it yields no unit.
    assert list(units) == ["front", "I", "II.A", "II.B"]
    change_logs = units["II.A"]
    assert change_logs.label == "Section II.A"
    assert change_logs.heading_path == (TITLE, "II. DISCUSSION", "A. Change Logs")
    assert change_logs.blocks[0].text == (
        "Placeholder recommendation: keep a log of each change to a record, "
        "with the operator and the earlier value."
    )
    assert change_logs.blocks[0].page == 3
    assert units["II.B"].blocks[0].page == 4
    # Banner, page numbers and the TOC page are not content.
    text = " ".join(b.text for u in units.values() for b in u.blocks)
    assert "Contains Nonbinding Recommendations" not in text
    assert "TABLE OF CONTENTS" not in text
    assert units["front"].blocks[0].text == "Synthetic Guidance on Ledgers January 2020"


def test_toc_nesting_follows_numbering_styles() -> None:
    pdf = PdfText(
        pages=(
            _toc(
                "I. Scope",
                "II. Approach",
                "A. Risk",
                "(1) Intended Use",
                "(2) Changes",
                "B. Records",
                "1. Wrapped entry that continues",
                "a. Sub question",
                "III. References",
                "Appendix A. Examples",
                "Example 1: Ledger System",
                "Example 2: Gadget System",
            ).replace("continues ........ 1", "continues\non the next line ........ 1"),
        )
    )
    entries = toc_headings(pdf, TOC, "g")
    shape = [(h.title, ".".join(n for _, n, _ in stack)) for h, stack in entries]
    assert shape == [
        ("I. Scope", "I"),
        ("II. Approach", "II"),
        ("A. Risk", "II.A"),
        ("(1) Intended Use", "II.A.1"),
        ("(2) Changes", "II.A.2"),
        ("B. Records", "II.B"),
        ("1. Wrapped entry that continues on the next line", "II.B.1"),
        ("a. Sub question", "II.B.1.a"),
        ("III. References", "III"),
        ("Appendix A. Examples", "A"),
        ("Example 1: Ledger System", "A.1"),
        ("Example 2: Gadget System", "A.2"),
    ]


@pytest.mark.parametrize(
    ("toc", "problem"),
    [
        (_toc("I. Scope", "III. Skipped"), "out of sequence"),
        (_toc("A. Starts with a letter", "C. Skips B"), "out of sequence"),
        (_toc("Introduction"), "has no section number"),
        ("TABLE OF CONTENTS\nI. Scope ..... 1\nII. Unfinished", "ends mid-entry"),
        ("TABLE OF CONTENTS\n", "no entries"),
    ],
)
def test_toc_problems_stop_the_parse(toc: str, problem: str) -> None:
    with pytest.raises(SourceParseError, match=problem):
        toc_headings(PdfText(pages=(toc,)), TOC, "g")


def test_headings_match_whole_lines_with_wrapping_and_footnote_markers() -> None:
    pdf = PdfText(
        pages=(
            _toc("I. Introduction", "II. A Heading That Wraps Across Two Lines"),
            "I. Introduction1\nBody with a time-\nstamped entry.\n"
            "The I. Introduction words inside a sentence are not a heading.\n"
            "II. A Heading That Wraps\nAcross Two Lines\nMore text.\n",
        )
    )
    units = _parse(pdf)
    assert [b.text for b in units["I"].blocks] == [
        "Body with a time-stamped entry. The I. Introduction words inside a sentence "
        "are not a heading."
    ]
    assert units["II"].blocks[0].text == "More text."


def test_missing_heading_stops_the_parse() -> None:
    pdf = PdfText(pages=(_toc("I. Scope"), "Some text\n"))
    with pytest.raises(SourceParseError, match="not found in the text"):
        _parse(pdf)


def test_paragraphs_and_pages_make_blocks_and_mid_page_numbers_are_kept() -> None:
    pdf = PdfText(
        pages=(
            _toc("I. Scope"),
            "7\nI. Scope\nFirst paragraph\ncontinues.\n \nCount\n21\nXI\nSecond.\n8\n",
            "9\nSame paragraph on the next page.\n",
        )
    )
    blocks = _parse(pdf)["I"].blocks
    assert [(b.page, b.text) for b in blocks] == [
        (2, "First paragraph continues."),
        (2, "Count 21 XI Second."),
        (3, "Same paragraph on the next page."),
    ]


def test_running_line_numbers_are_checked_and_removed() -> None:
    pdf = PdfText(
        pages=(
            _toc("I. Scope", "II. Next"),
            # Page number 1 at the top, then lines 1-6; footnotes follow line 6.
            " 1\nPreamble line 1 1\nI. Scope 2\nThe re-3\nexamination is 4\n 5\n"
            "(§ 11.10(e)) applies.2 6\n                    2 A footnote on this page.\n",
            # Line 6 repeated at the top of the next page is not a new line.
            " 2\n 6\nII. Next 7\nLast line. 8\n",
        )
    )
    units = _parse(pdf, PdfOptions(toc_pages=(1,), line_numbers_from_page=2))
    assert units["front"].blocks[0].text == "Preamble line 1"
    assert [b.text for b in units["I"].blocks] == [
        "The re-examination is",
        "(§ 11.10(e)) applies.2",
        "2 A footnote on this page.",
    ]
    assert units["II"].blocks[0].text == "Last line."


def test_a_numbered_page_without_its_next_line_number_stops_the_parse() -> None:
    pdf = PdfText(pages=(_toc("I. Scope"), "I. Scope 1\nText 2\n", "Unnumbered text\n"))
    with pytest.raises(SourceParseError, match="no line number 3"):
        _parse(pdf, PdfOptions(toc_pages=(1,), line_numbers_from_page=2))


def test_skipped_pages_are_not_content() -> None:
    pdf = PdfText(pages=(_toc("I. Scope"), "Contact details\n", "I. Scope\nText.\n"))
    assert "front" in _parse(pdf)
    assert "front" not in _parse(pdf, PdfOptions(toc_pages=(1,), skip_pages=(2,)))
