"""Parse an FDA guidance PDF into one unit per section.

Section headings come from the document's own table of contents. The TOC pages
named in the document's PdfOptions are read, dot leaders and page numbers removed,
and each entry's nesting taken from its numbering: "I." > "A." > "1." > "a.", plus
"(1)", "Appendix A" and "Example 1:". A number out of sequence stops the parse.
Each heading must then be found, in order, as whole lines of the body text; the
text up to the next heading belongs to it. Headings are never guessed: one that
cannot be found stops the parse.

Page furniture is removed: the "Contains Nonbinding Recommendations" banner, a bare
page number at the top or bottom of a page, and the TOC pages. In documents typeset
with running line numbers, each number is checked in sequence and removed; lines
after the last numbered line of a page are footnotes and are kept. Text before the
first heading becomes a "front matter" unit. Blocks are paragraphs (split at blank
lines) within one page, so every chunk knows the pages it came from.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from apps.rag.corpus import PdfOptions
from apps.rag.errors import SourceParseError
from apps.rag.parse_ecfr import ROMAN, ordinal
from apps.rag.units import Block, SourceUnit, match_key, normalize_space

FURNITURE = (
    re.compile(r"^Contains Nonbinding Recommendations$", re.IGNORECASE),
    re.compile(r"^Page \d+ of \d+$", re.IGNORECASE),
)
# A bare page number ("7", or "iii" in front matter) is furniture only at the top or
# bottom of its page; anywhere else it is content and is kept.
PAGE_NUMBER = re.compile(r"^(\d{1,3}|[ivxlc]{1,6})$")
_TOC_TITLE = re.compile(r"^table of contents$", re.IGNORECASE)
# Dot leaders, or a single spaced dot when the entry nearly fills the line, then
# the printed page number.
_TOC_LEADER = re.compile(r"(?:\s*\.{2,}[.\s]*|\s+\.\s+)\d+$")
_ENTRY = re.compile(
    r"^(?:Appendix\s+(?P<appendix>[A-Z])\b[.:]?"
    r"|Example\s+(?P<example>\d{1,2})\s*:"
    r"|\((?P<paren>\d{1,2})\)"
    r"|(?P<token>[IVXLC]{1,6}|[A-Z]|\d{1,2}|[a-z])\.)"
    r"\s+(?P<rest>\S.*)$"
)
# Numbering styles that make a section number like "III.C.2".
NUMBERED_STYLES = frozenset({"roman", "upper", "digit", "lower", "paren"})
# How many consecutive lines a heading may wrap across in the extracted text.
MAX_HEADING_LINES = 3


@dataclass(frozen=True)
class PdfText:
    pages: tuple[str, ...]


@dataclass(frozen=True)
class Heading:
    title: str  # as printed in the table of contents
    style: str
    number: str


@dataclass(frozen=True)
class _Line:
    page: int
    text: str
    new_paragraph: bool


def read_pdf(data: bytes) -> PdfText:
    try:
        reader = PdfReader(io.BytesIO(data))
        return PdfText(pages=tuple(page.extract_text() or "" for page in reader.pages))
    except PdfReadError as exc:
        raise SourceParseError(f"cannot read PDF: {exc}") from exc


# --- table of contents -----------------------------------------------------------


def _position(style: str, number: str) -> int | None:
    """Position of `number` in its style's sequence (1-based)."""
    if style == "roman":
        return ordinal(ROMAN, number.lower())
    if style in ("upper", "appendix"):
        return ord(number) - ord("A") + 1 if re.fullmatch(r"[A-Z]", number) else None
    if style == "lower":
        return ord(number) - ord("a") + 1 if re.fullmatch(r"[a-z]", number) else None
    return int(number) if number.isdigit() else None


Stack = list[tuple[str, str, str]]  # (style, number, heading title)


def _readings(options: list[tuple[str, str]], stack: Stack, title: str) -> list[Stack]:
    readings: list[Stack] = []
    for style, number in options:
        position = _position(style, number)
        if position is None:
            continue
        open_depths = [d for d, (s, _, _) in enumerate(stack) if s == style]
        if open_depths:
            depth = open_depths[-1]
            if position == (_position(style, stack[depth][1]) or 0) + 1:
                readings.append([*stack[:depth], (style, number, title)])
        elif position == 1:
            # Appendices sit at the top level, after the numbered sections.
            base = [] if style == "appendix" else stack
            readings.append([*base, (style, number, title)])
    return readings


def toc_headings(pdf: PdfText, options: PdfOptions, where: str) -> list[tuple[Heading, Stack]]:
    """Each TOC entry with the stack of headings it sits under (itself last)."""
    titles: list[str] = []
    pending = ""
    for page_number in options.toc_pages:
        if not 1 <= page_number <= len(pdf.pages):
            raise SourceParseError(f"{where}: TOC page {page_number} does not exist")
        for raw in pdf.pages[page_number - 1].splitlines():
            line = normalize_space(raw)
            if not line or _TOC_TITLE.match(line) or PAGE_NUMBER.match(line):
                continue
            if any(p.search(line) for p in FURNITURE):
                continue
            leader = _TOC_LEADER.search(line)
            if leader is None:  # an entry wrapped onto the next line
                pending = normalize_space(f"{pending} {line}")
                continue
            titles.append(normalize_space(f"{pending} {line[: leader.start()]}"))
            pending = ""
    if pending:
        raise SourceParseError(f"{where}: the table of contents ends mid-entry: {pending!r}")
    if not titles:
        raise SourceParseError(f"{where}: no entries found on TOC pages {options.toc_pages}")

    entries: list[tuple[Heading, Stack]] = []
    stack: Stack = []
    for title in titles:
        match = _ENTRY.match(title)
        if match is None:
            raise SourceParseError(f"{where}: TOC entry {title!r} has no section number")
        if match["appendix"]:
            options_ = [("appendix", match["appendix"])]
        elif match["example"]:
            options_ = [("example", match["example"])]
        elif match["paren"]:
            options_ = [("paren", match["paren"])]
        else:
            token = match["token"]
            options_ = [(s, token) for s in ("roman", "upper", "digit", "lower")]
        readings = _readings(options_, stack, title)
        if len(readings) != 1:
            state = " > ".join(t for _, _, t in stack) or "the start"
            problem = "is out of sequence" if not readings else "is ambiguous"
            raise SourceParseError(f"{where}: TOC entry {title!r} {problem} after {state}")
        stack = readings[0]
        style, number, _ = stack[-1]
        entries.append((Heading(title=title, style=style, number=number), stack))
    return entries


# --- page text -------------------------------------------------------------------


def _strip_page_numbers(page: list[str], *, top_only: bool) -> None:
    nonblank = [i for i, line in enumerate(page) if line]
    if nonblank and PAGE_NUMBER.match(page[nonblank[0]]):
        page[nonblank[0]] = ""
    nonblank = [i for i, line in enumerate(page) if line]
    if not top_only and nonblank and PAGE_NUMBER.match(page[nonblank[-1]]):
        page[nonblank[-1]] = ""


def content_lines(pdf: PdfText, options: PdfOptions, where: str) -> list[_Line]:
    lines: list[_Line] = []
    expected = 1  # next running line number, for line-numbered documents
    for number, text in enumerate(pdf.pages, start=1):
        if number in options.skip_pages or number in options.toc_pages:
            continue
        page = [normalize_space(raw) for raw in text.splitlines()]
        page = ["" if any(p.search(line) for p in FURNITURE) else line for line in page]
        numbered = (
            options.line_numbers_from_page is not None and number >= options.line_numbers_from_page
        )
        # A numbered page can end with a blank numbered line ("32"): only the top
        # line can be its page number.
        _strip_page_numbers(page, top_only=numbered)
        if not numbered:
            new_paragraph = True
            for line in page:
                if not line:
                    new_paragraph = True
                    continue
                lines.append(_Line(number, line, new_paragraph))
                new_paragraph = False
            continue

        new_paragraph, in_footnotes, numbered_lines = True, False, 0
        for line in page:
            if not line:
                continue
            if not in_footnotes and line == str(expected - 1):
                continue  # a blank numbered line repeated at the top of the next page
            if not in_footnotes:
                found = re.search(rf"(?<!\d){expected}$", line)
                if found is not None:
                    expected += 1
                    numbered_lines += 1
                    body = line[: found.start()].rstrip()
                    if not body:
                        new_paragraph = True  # a numbered blank line
                        continue
                    lines.append(_Line(number, body, new_paragraph))
                    new_paragraph = False
                    continue
                in_footnotes, new_paragraph = True, True
            lines.append(_Line(number, line, new_paragraph))
            new_paragraph = False
        if numbered_lines == 0 and any(page):
            raise SourceParseError(
                f"{where}: page {number} has text but no line number {expected}; "
                "check line_numbers_from_page"
            )
    return lines


def _locate(lines: list[_Line], start: int, title: str, where: str) -> tuple[int, int]:
    """(first line index, number of lines) of the heading in the body text.

    The heading must fill whole lines. A footnote marker printed after it
    ("I. Introduction1") is allowed.
    """
    target = match_key(title)
    for i in range(start, len(lines)):
        joined = ""
        for n in range(MAX_HEADING_LINES):
            if i + n >= len(lines):
                break
            joined += match_key(lines[i + n].text)
            if joined.startswith(target):
                extra = joined[len(target) :]
                if not extra or (extra.isdigit() and len(extra) <= 2):
                    return i, n + 1
            if not target.startswith(joined):
                break
    raise SourceParseError(f"{where}: TOC heading {title!r} not found in the text")


def _join_lines(texts: list[str]) -> str:
    out = ""
    for text in texts:
        if not out:
            out = text
        elif out.endswith("-") and text[:1].islower():
            out += text  # a word hyphenated across lines: "time-" + "stamped"
        else:
            out += " " + text
    return out


def _blocks(lines: list[_Line]) -> tuple[Block, ...]:
    """One block per paragraph per page."""
    blocks: list[Block] = []
    current: list[_Line] = []
    for line in [*lines, None]:
        if line is None or (current and (line.page != current[0].page or line.new_paragraph)):
            if current:
                blocks.append(
                    Block(text=_join_lines([c.text for c in current]), page=current[0].page)
                )
            current = []
        if line is not None:
            current.append(line)
    return tuple(blocks)


def _locator_part(style: str, number: str) -> str:
    if style == "appendix":
        return f"appendix-{number.lower()}"
    if style == "example":
        return f"example-{number}"
    return number


def parse_guidance(
    pdf: PdfText, *, doc_id: str, title: str, options: PdfOptions
) -> list[SourceUnit]:
    headings = toc_headings(pdf, options, doc_id)
    lines = content_lines(pdf, options, doc_id)
    positions: list[tuple[int, int]] = []
    cursor = 0
    for heading, _ in headings:
        index, height = _locate(lines, cursor, heading.title, doc_id)
        positions.append((index, height))
        cursor = index + height

    units: list[SourceUnit] = []
    front = lines[: positions[0][0]]
    if front:
        units.append(
            SourceUnit(
                doc_id=doc_id,
                locator="front",
                label="Front matter",
                heading_path=(title,),
                blocks=_blocks(front),
            )
        )
    for k, (heading, stack) in enumerate(headings):
        index, height = positions[k]
        end = positions[k + 1][0] if k + 1 < len(positions) else len(lines)
        body = lines[index + height : end]
        if not body:
            continue  # a heading directly followed by its first subheading
        locator = ".".join(_locator_part(s, n) for s, n, _ in stack)
        numbered = all(s in NUMBERED_STYLES for s, _, _ in stack)
        units.append(
            SourceUnit(
                doc_id=doc_id,
                locator=locator,
                label=f"Section {locator}" if numbered else heading.title,
                heading_path=(title, *(t for _, _, t in stack)),
                blocks=_blocks(body),
            )
        )
    return units
