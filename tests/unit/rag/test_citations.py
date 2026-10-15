"""Section citations in questions, and which chunks they point at."""

from __future__ import annotations

import pytest

from apps.rag.citations import Citation, parse_citations, points_at
from apps.rag.corpus import SourceKind

REG: SourceKind = "regulation"
GUIDE: SourceKind = "guidance"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What does 21 CFR 11.10(e) require?", [(REG, "11.10(e)")]),
        ("How does §11.3 define a closed system?", [(REG, "11.3")]),
        ("Under 11.200(a)(1)(i), and section 11.1?", [(REG, "11.200(a)(1)(i)"), (REG, "11.1")]),
        ("Compare §§ 11.30 and 11.10(e), then 11.30 again.", [(REG, "11.30"), (REG, "11.10(e)")]),
        ("What do sections III.C.4 and V.A say?", [(GUIDE, "III.C.4"), (GUIDE, "V.A")]),
        ("See III.1.a (and 11.50).", [(GUIDE, "III.1.a"), (REG, "11.50")]),
    ],
)
def test_citations_are_found_in_order_without_repeats(
    question: str, expected: list[tuple[str, str]]
) -> None:
    assert [(c.kind, c.locator) for c in parse_citations(question)] == expected


@pytest.mark.parametrize(
    "question",
    [
        "Does Part 11 apply to a system installed in August 1997?",
        "Version 1.2.3 of v1.11.10 scored 0.8.",
        "I.e. what does section III say?",
        "Is V.A.6s a section?",
    ],
)
def test_things_that_are_not_citations(question: str) -> None:
    assert parse_citations(question) == []


@pytest.mark.parametrize(
    ("citation", "kind", "locator", "expected"),
    [
        (Citation(REG, "11.10"), REG, "11.10(e)", True),
        (Citation(REG, "11.10"), REG, "11.10-intro", True),
        (Citation(REG, "11.10(e)"), REG, "11.10(e)", True),
        (Citation(REG, "11.10(e)(1)"), REG, "11.10(e)", True),  # the chunk holds (e)(1)
        (Citation(REG, "11.30"), REG, "11.30", True),
        (Citation(REG, "11.1"), REG, "11.10(a)", False),
        (Citation(REG, "11.3"), REG, "11.30", False),
        (Citation(REG, "11.10(e)"), REG, "11.10-intro", False),
        (Citation(REG, "11.10(e)"), REG, "11.10(f)", False),
        (Citation(REG, "11.10"), GUIDE, "11.10", False),
        (Citation(GUIDE, "III.C"), GUIDE, "III.C.4", True),
        (Citation(GUIDE, "III.C.4.a"), GUIDE, "III.C.4", True),
        (Citation(GUIDE, "III.C.4"), GUIDE, "III.C.4", True),
        (Citation(GUIDE, "III.1"), GUIDE, "III.10", False),
        (Citation(GUIDE, "III.C"), GUIDE, "III.C-intro", False),
        (Citation(GUIDE, "III.C.4"), REG, "III.C.4", False),
    ],
)
def test_which_chunks_a_citation_points_at(
    citation: Citation, kind: str, locator: str, expected: bool
) -> None:
    assert points_at(citation, kind, locator) is expected
