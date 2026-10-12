"""Guidance PDF parsing on generated PDFs: outline sections, furniture, page tracking."""

from __future__ import annotations

import pytest
from pypdf.generic import Destination

from apps.rag.corpus import PdfOptions
from apps.rag.errors import SourceParseError
from apps.rag.parse_guidance import OutlineEntry, PdfText, parse_guidance, read_pdf
from apps.rag.units import SourceUnit
from tests.rag_helpers import GUIDANCE_OUTLINE, GUIDANCE_PAGES, make_pdf

TITLE = "Synthetic Guidance on Ledgers (test fixture)"


def _parse(pdf: PdfText, options: PdfOptions | None = None) -> dict[str, SourceUnit]:
    units = parse_guidance(pdf, doc_id="g", title=TITLE, options=options)
    return {u.locator: u for u in units}


def test_read_pdf_returns_page_text_and_nested_outline() -> None:
    pdf = read_pdf(make_pdf(GUIDANCE_PAGES, GUIDANCE_OUTLINE))
    assert len(pdf.pages) == 3
    assert "A. Change Logs" in pdf.pages[1]
    assert pdf.outline == (
        OutlineEntry(1, "I. INTRODUCTION", 2),
        OutlineEntry(1, "II. DISCUSSION", 2),
        OutlineEntry(2, "A. Change Logs", 2),
        OutlineEntry(2, "B. Passwords", 3),
    )


def test_sections_follow_the_outline() -> None:
    units = _parse(read_pdf(make_pdf(GUIDANCE_PAGES, GUIDANCE_OUTLINE)))
    # "II. DISCUSSION" has no text of its own before "A.", so it yields no unit.
    assert list(units) == ["front", "I", "II.A", "II.B"]
    change_logs = units["II.A"]
    assert change_logs.label == "Section II.A"
    assert change_logs.heading_path == (TITLE, "II. DISCUSSION", "A. Change Logs")
    assert change_logs.blocks[0].text == (
        "Placeholder recommendation: keep a log of each change to a record, "
        "with the operator and the earlier value."
    )
    assert change_logs.blocks[0].page == 2
    assert units["II.B"].blocks[0].page == 3


def test_page_furniture_is_removed() -> None:
    units = _parse(read_pdf(make_pdf(GUIDANCE_PAGES, GUIDANCE_OUTLINE)))
    text = " ".join(b.text for u in units.values() for b in u.blocks)
    assert "Contains Nonbinding Recommendations" not in text
    assert units["front"].blocks[0].text == "Synthetic Guidance on Ledgers January 2020"
    assert units["II.A"].blocks[0].text.endswith("earlier value.")  # page number "2" dropped


def test_numbers_and_numeral_words_inside_a_page_are_kept() -> None:
    pdf = PdfText(
        pages=("I. SCOPE\nCount\n21\nXI\ncivil\nEnd of scope.\n4\n",),
        outline=(OutlineEntry(1, "I. SCOPE", 1),),
    )
    # Only the trailing "4" is a page number.
    assert _parse(pdf)["I"].blocks[0].text == "Count 21 XI civil End of scope."


def test_section_spanning_pages_keeps_one_block_per_page() -> None:
    pdf = PdfText(
        pages=(
            "I. SCOPE\nFirst page text of the scope section\n",
            "continues on the second page.\nII. OTHER\nOther text.",
        ),
        outline=(OutlineEntry(1, "I. SCOPE", 1), OutlineEntry(1, "II. OTHER", 2)),
    )
    blocks = _parse(pdf)["I"].blocks
    assert [(b.page, b.text) for b in blocks] == [
        (1, "First page text of the scope section"),
        (2, "continues on the second page."),
    ]


def test_wrapped_headings_unnumbered_sections_and_hyphenation() -> None:
    pdf = PdfText(
        pages=(
            "1. A Heading That Wraps\nAcross Two Lines\nBody with a time-\nstamped entry.\n"
            "Glossary\nTerm list.\n",
        ),
        outline=(
            OutlineEntry(1, "1. A Heading That Wraps Across Two Lines", 1),
            OutlineEntry(1, "Glossary", 1),
        ),
    )
    units = _parse(pdf)
    assert units["1"].blocks[0].text == "Body with a time-stamped entry."
    assert units["glossary"].label == "Glossary"


def test_table_of_contents_lines_and_skipped_pages() -> None:
    pdf = PdfText(
        pages=("TABLE OF CONTENTS\nI. SCOPE ........ 2\n", "I. SCOPE\nScope text.\n"),
        outline=(OutlineEntry(1, "I. SCOPE", 2),),
    )
    assert [b.text for b in _parse(pdf)["front"].blocks] == ["TABLE OF CONTENTS"]
    assert "front" not in _parse(pdf, PdfOptions(skip_pages=(1,)))


def test_outline_entry_without_a_page_stops_the_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    from pypdf import PdfReader

    data = make_pdf(GUIDANCE_PAGES, GUIDANCE_OUTLINE)
    original = PdfReader.get_destination_page_number

    def lose_one_page(reader: PdfReader, destination: Destination) -> int | None:
        if destination.title == "B. Passwords":
            return -1
        return original(reader, destination)

    monkeypatch.setattr(PdfReader, "get_destination_page_number", lose_one_page)
    with pytest.raises(SourceParseError, match=r"'B\. Passwords' has no title or no page"):
        read_pdf(data)


def test_no_outline_stops_the_parse() -> None:
    with pytest.raises(SourceParseError, match="no outline"):
        _parse(PdfText(pages=("I. SCOPE\nText",), outline=()))


def test_outline_heading_missing_from_text_stops_the_parse() -> None:
    pdf = PdfText(pages=("Some text\n",), outline=(OutlineEntry(1, "I. SCOPE", 1),))
    with pytest.raises(SourceParseError, match="not found in the text"):
        _parse(pdf)


def test_duplicate_section_locators_stop_the_parse() -> None:
    pdf = PdfText(
        pages=("A. One\nx\nA. Two\ny\n",),
        outline=(OutlineEntry(1, "A. One", 1), OutlineEntry(1, "A. Two", 1)),
    )
    with pytest.raises(SourceParseError, match="locator 'A'"):
        _parse(pdf)
