"""Parse an FDA guidance PDF into one unit per section.

Section boundaries come from the PDF's outline (bookmarks), which FDA guidance PDFs
generate from their headings. Each outline entry is located in the page text, and
the text up to the next entry belongs to it. A document without an outline, or an
outline entry that cannot be found in the text, stops the parse: headings are not
guessed from text patterns.

Lines repeated as page furniture (the "Contains Nonbinding Recommendations" banner,
page numbers, table-of-contents leaders) are removed. Text before the first heading
becomes a "front matter" unit. Each page's share of a section is kept as its own
block, so every chunk knows the pages it came from.
"""

from __future__ import annotations

import io
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from pypdf.generic import Destination

from apps.rag.corpus import PdfOptions
from apps.rag.errors import SourceParseError
from apps.rag.units import Block, SourceUnit, match_key, normalize_space

FURNITURE = (
    re.compile(r"^Contains Nonbinding Recommendations$", re.IGNORECASE),
    re.compile(r"^Page \d+ of \d+$", re.IGNORECASE),
    re.compile(r"\.{5,}\s*\d+$"),  # table-of-contents leader lines
)
# A bare page number ("7", or "iii" in front matter) is furniture only as the first
# or last line of its page; anywhere else it is content and is kept.
PAGE_NUMBER = re.compile(r"^(\d{1,3}|[ivxlc]{1,6})$")
_NUMBERING = re.compile(r"^(?P<num>[IVXLC]+|[A-Z]|\d{1,2}|[a-z])[.)]\s+(?P<rest>\S.*)$")
# How many consecutive lines a heading may wrap across in the extracted text.
MAX_HEADING_LINES = 3


@dataclass(frozen=True)
class OutlineEntry:
    level: int  # 1 = top level
    title: str
    page: int  # 1-based


@dataclass(frozen=True)
class PdfText:
    pages: tuple[str, ...]
    outline: tuple[OutlineEntry, ...]


@dataclass(frozen=True)
class _Line:
    page: int
    text: str


def read_pdf(data: bytes) -> PdfText:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = tuple(page.extract_text() or "" for page in reader.pages)
        entries: list[OutlineEntry] = []

        def walk(items: Sequence[object], level: int) -> None:
            for item in items:
                if isinstance(item, list):
                    walk(item, level + 1)
                elif isinstance(item, Destination):
                    number = reader.get_destination_page_number(item)
                    title = normalize_space(str(item.title or ""))
                    if not title or number is None or number < 0:
                        # Skipping it would merge its text into the previous section.
                        raise SourceParseError(
                            f"outline entry {title!r} has no title or no page it points to"
                        )
                    entries.append(OutlineEntry(level, title, number + 1))

        walk(reader.outline, 1)
    except PdfReadError as exc:
        raise SourceParseError(f"cannot read PDF: {exc}") from exc
    return PdfText(pages=pages, outline=tuple(entries))


def content_lines(pdf: PdfText, options: PdfOptions) -> list[_Line]:
    lines: list[_Line] = []
    for number, text in enumerate(pdf.pages, start=1):
        if number in options.skip_pages:
            continue
        page = [normalize_space(raw) for raw in text.splitlines()]
        page = [line for line in page if line and not any(p.search(line) for p in FURNITURE)]
        if page and PAGE_NUMBER.match(page[-1]):
            page.pop()
        if page and PAGE_NUMBER.match(page[0]):
            page.pop(0)
        lines += [_Line(number, line) for line in page]
    return lines


def _locate(lines: list[_Line], start: int, entry: OutlineEntry, where: str) -> tuple[int, int]:
    """(first line index, number of lines) of the heading text for `entry`."""
    target = match_key(entry.title)
    for i in range(start, len(lines)):
        if lines[i].page < entry.page - 1:
            continue
        if lines[i].page > entry.page + 1:
            break
        joined = ""
        for n in range(MAX_HEADING_LINES):
            if i + n >= len(lines):
                break
            joined += match_key(lines[i + n].text)
            if joined == target:
                return i, n + 1
            if not target.startswith(joined):
                break
    raise SourceParseError(
        f"{where}: outline heading {entry.title!r} (page {entry.page}) not found in the text"
    )


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
    blocks: list[Block] = []
    page_lines: list[str] = []
    page: int | None = None
    for line in [*lines, None]:
        if line is None or line.page != page:
            if page_lines and page is not None:
                blocks.append(Block(text=_join_lines(page_lines), page=page))
            page_lines = []
            if line is None:
                break
            page = line.page
        page_lines.append(line.text)
    return tuple(blocks)


def _number_and_title(heading: str) -> tuple[str | None, str]:
    match = _NUMBERING.match(heading)
    if match is None:
        return None, heading
    return match.group("num"), match.group("rest")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")[:40] or "section"


def parse_guidance(
    pdf: PdfText, *, doc_id: str, title: str, options: PdfOptions | None = None
) -> list[SourceUnit]:
    if not pdf.outline:
        raise SourceParseError(
            f"{doc_id}: the PDF has no outline; section boundaries would have to be guessed"
        )
    lines = content_lines(pdf, options or PdfOptions())
    # Locate every heading, in outline order.
    positions: list[tuple[int, int]] = []
    cursor = 0
    for entry in pdf.outline:
        index, height = _locate(lines, cursor, entry, doc_id)
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
    ancestors: list[tuple[int, str, str]] = []  # (level, heading, locator part)
    seen: set[str] = set()
    for k, entry in enumerate(pdf.outline):
        ancestors = [a for a in ancestors if a[0] < entry.level]
        number, _ = _number_and_title(entry.title)
        ancestors.append((entry.level, entry.title, number or _slug(entry.title)))
        locator = ".".join(part for _, _, part in ancestors)
        if locator in seen:
            raise SourceParseError(f"{doc_id}: two sections resolve to locator {locator!r}")
        seen.add(locator)
        index, height = positions[k]
        end = positions[k + 1][0] if k + 1 < len(positions) else len(lines)
        body = lines[index + height : end]
        if not body:
            continue  # a heading directly followed by its first subheading
        numbered = all(_number_and_title(h)[0] for _, h, _ in ancestors)
        units.append(
            SourceUnit(
                doc_id=doc_id,
                locator=locator,
                label=f"Section {locator}" if numbered else entry.title,
                heading_path=(title, *(h for _, h, _ in ancestors)),
                blocks=_blocks(body),
            )
        )
    return units
