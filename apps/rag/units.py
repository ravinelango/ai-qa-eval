"""Structural pieces of a source document, as produced by the parsers.

A `SourceUnit` is the smallest structural division the corpus is chunked by: one
top-level paragraph of a CFR section (with its subparagraphs), a section's
introductory text, or one section of a guidance document. The chunker turns each
unit into one chunk, or several when the unit does not fit the embedding model's
token limit.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Block(_Strict):
    """One run of source text and where it came from."""

    text: str = Field(min_length=1)
    # Full paragraph label, e.g. "§11.50(a)(1)"; None for unlabeled text.
    paragraph: str | None = None
    # 1-based PDF page, for guidance.
    page: int | None = Field(default=None, ge=1)
    # Element locator inside the source file, for eCFR XML (e.g. "DIV8[§ 11.10]/P[6]").
    element: str | None = None


class SourceUnit(_Strict):
    doc_id: str
    # Stable within the document; becomes part of the chunk ID.
    locator: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.()\-]*$")
    # How a reader would cite it: "§11.10(e)", "§11.10 (introductory text)", "Section III.C.2".
    label: str
    # Enclosing headings, outermost first, ending with the unit's own heading.
    heading_path: tuple[str, ...]
    blocks: tuple[Block, ...] = Field(min_length=1)


def normalize_space(text: str) -> str:
    """Collapse all whitespace (including no-break spaces) to single spaces."""
    return " ".join(text.split())


def match_key(text: str) -> str:
    """Case- and punctuation-insensitive form used to match headings."""
    return re.sub(r"[^a-z0-9]+", "", text.casefold())
