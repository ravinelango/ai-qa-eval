"""Section citations in a question, and the chunks they point at.

A question that cites a section ("21 CFR 11.10(e)", "§ 11.3", "section III.C.4 of the
guidance") is asking about that section, so in hybrid mode the retriever ranks the
chunks it points at first (apps.rag.retrieve). Two forms are recognized:

- Regulation: <part>.<section>, with any paragraph designators, e.g. 11.10, 11.10(e)
  or 11.200(a)(1). It points at regulation chunks of that part and section. A title
  number in front ("21 CFR") is not checked; the corpus holds one CFR title.
- Guidance: a Roman numeral and at least one more component, e.g. III.C.4, V.A or
  III.1.a. It points at chunks with that section number in any guidance document.
  A Roman numeral alone ("section III") is not recognized.

A citation points at a chunk when the chunk's locator is the cited one or sits under
it (11.10 covers 11.10(e) and 11.10-intro; III.C covers III.C.4). A regulation chunk
also covers a finer citation inside it: the chunk 11.10(e) holds 11.10(e)(1), because
regulation chunks are whole top-level paragraphs. Guidance chunks do not: every
numbered heading in a guidance document is its own chunk, so a guidance section with
no subsections (section III of one document) does not hold III.C.4 (of another).
Nothing is inferred from document titles or question numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from apps.rag.corpus import SourceKind

_REGULATION = re.compile(r"(?<![\w.])([1-9]\d{0,2}\.\d{1,4}(?:\([a-z0-9]{1,5}\))*+)(?!\w|\.\d)")
_GUIDANCE = re.compile(r"(?<![\w.])([IVX]+\.(?:[A-Z]|\d{1,3})(?:\.(?:[A-Za-z]|\d{1,3}))*+)(?!\w)")
_PATTERNS: tuple[tuple[SourceKind, re.Pattern[str]], ...] = (
    ("regulation", _REGULATION),
    ("guidance", _GUIDANCE),
)


@dataclass(frozen=True)
class Citation:
    kind: SourceKind
    locator: str


def parse_citations(question: str) -> list[Citation]:
    """Citations in the order they first appear, without repeats."""
    found: list[tuple[int, Citation]] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(question):
            found.append((match.start(), Citation(kind, match.group(1))))
    out: list[Citation] = []
    for _, citation in sorted(found, key=lambda f: f[0]):
        if citation not in out:
            out.append(citation)
    return out


def points_at(citation: Citation, doc_kind: str, locator: str) -> bool:
    if doc_kind != citation.kind:
        return False
    want = citation.locator
    if citation.kind == "guidance":
        return locator == want or locator.startswith(want + ".")
    return (
        locator in (want, f"{want}-intro")
        or locator.startswith(want + "(")
        or want.startswith(locator + "(")
    )
