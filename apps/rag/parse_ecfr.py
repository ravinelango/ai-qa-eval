"""Parse one CFR part from eCFR XML into paragraph-level units.

The eCFR XML nests a part (DIV5 TYPE="PART") into optional subparts (DIV6) and
sections (DIV8). Paragraphs are <P> elements whose text starts with markers such
as "(a)", "(1)", "(i)" or "(A)". Their nesting is not marked up, so it is inferred
from the marker sequence: "(i)" after "(h)" is paragraph (i), but "(i)" after "(1)"
opens a roman-numeral level.

Each top-level paragraph, with all its subparagraphs, becomes one unit labeled like
"§11.10(e)". Text before a section's first paragraph becomes "§11.10 (introductory
text)"; a section with no paragraph markers is a single unit "§11.30".

Elements the parser does not recognize stop the parse instead of being dropped.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as SafeET

from apps.rag.errors import SourceParseError
from apps.rag.units import Block, SourceUnit, normalize_space

PARAGRAPH_TAGS = frozenset({"P", "FP"})
# Present in eCFR XML but not regulatory text: authority and source citations,
# and the Federal Register citation trailer of each section.
IGNORED_TAGS = frozenset({"AUTH", "SOURCE", "CITA", "SECAUTH"})

_SECTION_HEAD = re.compile(r"^§\s*(?P<number>\d+\.\d+[a-z]?)\s+(?P<subject>.+)$")
_MARKER = re.compile(r"^\(([a-z]{1,2}|[A-Z]|\d{1,3}|[ivxlc]{1,7})\)\s*")
_ROMAN = re.compile(r"^(c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}

# Paragraph levels, outermost first: (a), (1), (i), (A).
LETTER, DIGIT, ROMAN, UPPER = 1, 2, 3, 4


def _roman_value(marker: str) -> int | None:
    if not marker or not _ROMAN.match(marker):
        return None
    total = 0
    for i, ch in enumerate(marker):
        value = _ROMAN_VALUES[ch]
        following = _ROMAN_VALUES[marker[i + 1]] if i + 1 < len(marker) else 0
        total += -value if value < following else value
    return total


def ordinal(level: int, marker: str) -> int | None:
    """Position of `marker` in the sequence of `level` (1-based), or None if invalid."""
    if level == LETTER and re.fullmatch(r"[a-z]{1,2}", marker):
        if len(marker) == 2 and marker[0] != marker[1]:
            return None  # the CFR continues z with aa, bb, cc
        return (ord(marker[0]) - ord("a") + 1) + 26 * (len(marker) - 1)
    if level == DIGIT and marker.isdigit():
        return int(marker)
    if level == ROMAN:
        return _roman_value(marker)
    if level == UPPER and re.fullmatch(r"[A-Z]", marker):
        return ord(marker) - ord("A") + 1
    return None


Stack = list[tuple[int, str]]


def _readings(marker: str, stack: Stack) -> list[Stack]:
    """Every stack `marker` could produce: continuing an open level, or opening the next."""
    readings: list[Stack] = []
    for depth in range(len(stack) - 1, -1, -1):
        level, current = stack[depth]
        position, previous = ordinal(level, marker), ordinal(level, current)
        if position is not None and previous is not None and position == previous + 1:
            readings.append([*stack[:depth], (level, marker)])
    deeper = stack[-1][0] + 1 if stack else LETTER
    if ordinal(deeper, marker) == 1:
        readings.append([*stack, (deeper, marker)])
    return readings


def assign_level(marker: str, stack: Stack, where: str, next_marker: str | None = None) -> Stack:
    """The paragraph stack after `marker`.

    "(i)" after "(h)(1)" could be paragraph (i) or (h)(1)(i). Such a marker is resolved
    by the marker that follows it; if that does not settle it, the parse stops.
    """
    sequence = "".join(f"({m})" for _, m in stack) or "start of section"
    readings = _readings(marker, stack)
    if not readings:
        raise SourceParseError(f"{where}: paragraph ({marker}) does not follow {sequence}")
    if len(readings) > 1 and next_marker is not None:
        readings = [r for r in readings if _readings(next_marker, r)]
    if len(readings) != 1:
        raise SourceParseError(f"{where}: paragraph ({marker}) after {sequence} is ambiguous")
    return readings[0]


def leading_markers(text: str) -> tuple[list[str], str]:
    markers: list[str] = []
    rest = text
    while match := _MARKER.match(rest):
        markers.append(match.group(1))
        rest = rest[match.end() :]
    return markers, rest


def element_text(element: Element) -> str:
    return normalize_space("".join(element.itertext()))


def _children(element: Element) -> Iterator[Element]:
    yield from list(element)


def _find_part(root: Element, part: str) -> Element:
    parts = [e for e in root.iter() if e.get("TYPE") == "PART" and e.get("N") == part]
    if len(parts) != 1:
        raise SourceParseError(f"expected exactly one PART {part} in the XML, found {len(parts)}")
    return parts[0]


def _head(element: Element, where: str) -> str:
    head = element.find("HEAD")
    if head is None:
        raise SourceParseError(f"{where}: no HEAD element")
    return element_text(head)


def parse_ecfr_part(data: bytes, *, doc_id: str, part: str) -> list[SourceUnit]:
    try:
        root = SafeET.fromstring(data)
    except SafeET.ParseError as exc:
        raise SourceParseError(f"{doc_id}: not well-formed XML: {exc}") from exc
    part_el = _find_part(root, part)
    part_head = _head(part_el, f"PART {part}")
    units: list[SourceUnit] = []
    for child in _children(part_el):
        if child.tag == "HEAD" or child.tag in IGNORED_TAGS:
            continue
        if child.tag == "DIV6" and child.get("TYPE") == "SUBPART":
            subpart_head = _head(child, f"subpart {child.get('N')}")
            for section in _children(child):
                if section.tag == "HEAD" or section.tag in IGNORED_TAGS:
                    continue
                units += _parse_section(section, doc_id, (part_head, subpart_head))
        else:
            units += _parse_section(child, doc_id, (part_head,))
    if not units:
        raise SourceParseError(f"{doc_id}: PART {part} contains no sections")
    return units


def _parse_section(element: Element, doc_id: str, parents: tuple[str, ...]) -> list[SourceUnit]:
    if element.tag != "DIV8" or element.get("TYPE") != "SECTION":
        raise SourceParseError(
            f"{doc_id}: unexpected element <{element.tag} TYPE={element.get('TYPE')!r}> "
            "where a section was expected"
        )
    head = _head(element, f"section {element.get('N')}")
    match = _SECTION_HEAD.match(head)
    if match is None:
        raise SourceParseError(f"{doc_id}: cannot read a section number from {head!r}")
    number = match.group("number")
    where = f"{doc_id} §{number}"
    heading_path = (*parents, head)
    node = f"DIV8[{element.get('N') or number}]"

    # (element locator, text, leading markers) for every paragraph, in order.
    paragraph_elements: list[tuple[str, str, list[str]]] = []
    p_index = 0
    for child in _children(element):
        if child.tag == "HEAD" or child.tag in IGNORED_TAGS:
            continue
        if child.tag not in PARAGRAPH_TAGS and not child.tag.startswith("FP-"):
            raise SourceParseError(f"{where}: unexpected element <{child.tag}> in a section")
        p_index += 1
        text = element_text(child)
        if text:
            locator = f"{node}/{child.tag}[{p_index}]"
            paragraph_elements.append((locator, text, leading_markers(text)[0]))
    all_markers = [m for _, _, markers in paragraph_elements for m in markers]

    intro: list[Block] = []
    paragraphs: list[tuple[str, list[Block]]] = []  # (top-level marker, blocks)
    stack: Stack = []
    seen = 0  # markers consumed so far, for looking one ahead
    for locator, text, markers in paragraph_elements:
        top_before = stack[0] if stack else None
        for marker in markers:
            seen += 1
            following = all_markers[seen] if seen < len(all_markers) else None
            stack = assign_level(marker, stack, where, following)
        if not stack:
            intro.append(Block(text=text, element=locator))
            continue
        if stack[0] != top_before:  # a new top-level paragraph, e.g. "(b)" or "(b)(1)"
            paragraphs.append((stack[0][1], []))
        label = f"§{number}" + "".join(f"({m})" for _, m in stack)
        paragraphs[-1][1].append(Block(text=text, paragraph=label, element=locator))

    if not intro and not paragraphs:
        raise SourceParseError(f"{where}: section has no text")
    units: list[SourceUnit] = []
    if intro:
        if paragraphs:
            locator, label = f"{number}-intro", f"§{number} (introductory text)"
        else:
            locator, label = number, f"§{number}"
        units.append(
            SourceUnit(
                doc_id=doc_id,
                locator=locator,
                label=label,
                heading_path=heading_path,
                blocks=tuple(intro),
            )
        )
    for top, blocks in paragraphs:
        units.append(
            SourceUnit(
                doc_id=doc_id,
                locator=f"{number}({top})",
                label=f"§{number}({top})",
                heading_path=heading_path,
                blocks=tuple(blocks),
            )
        )
    return units
