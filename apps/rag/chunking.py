"""Turn structural units into chunks that fit the embedding model, and save the
chunk manifest.

What is embedded for a chunk is its heading path, its label and its text, one per
line (`Chunk.embed_text`). The token count of that whole string, special tokens
included, must not exceed the model's limit. A unit that does not fit is split into
parts: first between its blocks (paragraphs, or a guidance section's pages), then
between sentences, then between words. Every part keeps the unit's heading path and
records the paragraph labels, pages and source elements it covers. Each finished
chunk is counted again, and one over the limit is an error; nothing is truncated.

Chunk IDs are `<doc_id>/<locator>`, with `#<part>` appended when a unit is split.
They depend only on the document structure, so the same inputs and configuration
always give the same IDs, the same chunks and a byte-identical manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from apps.rag.corpus import SourceDocument, SourceKind
from apps.rag.errors import ChunkingError
from apps.rag.units import Block, SourceUnit

_SENTENCE_BREAK = re.compile(r"(?<=[.;:?!])\s+")
# Placeholder used while packing parts; the real "(part k of n)" is counted again after.
_PART_PLACEHOLDER = " (part 99 of 99)"


class TokenCounter(Protocol):
    """Counts tokens exactly as the embedding model will see them."""

    @property
    def max_tokens(self) -> int: ...

    def count(self, text: str) -> int:
        """Token count including the model's special tokens; never truncates."""
        ...


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Chunk(_Strict):
    id: str
    doc_id: str
    doc_title: str
    doc_kind: SourceKind
    domain: str
    locator: str
    label: str
    heading_path: tuple[str, ...]
    paragraphs: tuple[str, ...]
    pages: tuple[int, ...]
    elements: tuple[str, ...]
    part: int = Field(ge=1)
    parts: int = Field(ge=1)
    text: str = Field(min_length=1)
    token_count: int = Field(ge=1)
    text_sha256: str

    @property
    def embed_text(self) -> str:
        return compose_embed_text(self.heading_path, self.label, self.text)


def compose_embed_text(heading_path: Sequence[str], label: str, text: str) -> str:
    return "\n".join([*heading_path, label, text])


class ChunkManifest(_Strict):
    schema_version: Literal[1] = 1
    chunker_version: int
    corpus_fingerprint: str
    tokenizer_fingerprint: str
    max_tokens: int
    chunk_count: int
    chunks: tuple[Chunk, ...]


# --- splitting -----------------------------------------------------------------


_Piece = tuple[int, str]  # (index of the source block, text)


def _fits(counter: TokenCounter, unit: SourceUnit, label: str, texts: Sequence[str]) -> bool:
    text = "\n".join(texts)
    return counter.count(compose_embed_text(unit.heading_path, label, text)) <= counter.max_tokens


def _split_oversized(
    counter: TokenCounter, unit: SourceUnit, label: str, index: int, text: str
) -> list[_Piece]:
    """Split one block's text into pieces that each fit on their own."""
    pieces: list[_Piece] = []
    for sentence in _SENTENCE_BREAK.split(text):
        if _fits(counter, unit, label, [sentence]):
            pieces.append((index, sentence))
            continue
        words = sentence.split(" ")
        current: list[str] = []
        for word in words:
            if _fits(counter, unit, label, [" ".join([*current, word])]):
                current.append(word)
                continue
            if not current:
                raise ChunkingError(
                    f"{unit.doc_id}/{unit.locator}: a single word does not fit within "
                    f"{counter.max_tokens} tokens with its headings"
                )
            pieces.append((index, " ".join(current)))
            current = [word]
        if current:
            pieces.append((index, " ".join(current)))
    return pieces


def _pack(
    counter: TokenCounter, unit: SourceUnit, label: str, pieces: list[_Piece]
) -> list[list[_Piece]]:
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    for piece in pieces:
        candidate = [*current, piece]
        if not current or _fits(counter, unit, label, _texts(candidate)):
            current = candidate
        else:
            groups.append(current)
            current = [piece]
    if current:
        groups.append(current)
    return groups


def _texts(pieces: Sequence[_Piece]) -> list[str]:
    """Pieces from the same block are joined by spaces, blocks by newlines."""
    lines: list[str] = []
    previous: int | None = None
    for index, text in pieces:
        if index == previous:
            lines[-1] += " " + text
        else:
            lines.append(text)
        previous = index
    return lines


def _ordered_unique[T](values: Iterable[T | None]) -> tuple[T, ...]:
    out: list[T] = []
    for value in values:
        if value is not None and value not in out:
            out.append(value)
    return tuple(out)


def chunk_unit(unit: SourceUnit, document: SourceDocument, counter: TokenCounter) -> list[Chunk]:
    whole = [b.text for b in unit.blocks]
    if _fits(counter, unit, unit.label, whole):
        groups = [[(i, b.text) for i, b in enumerate(unit.blocks)]]
    else:
        packing_label = unit.label + _PART_PLACEHOLDER
        pieces: list[_Piece] = []
        for i, block in enumerate(unit.blocks):
            if _fits(counter, unit, packing_label, [block.text]):
                pieces.append((i, block.text))
            else:
                pieces += _split_oversized(counter, unit, packing_label, i, block.text)
        groups = _pack(counter, unit, packing_label, pieces)

    chunks: list[Chunk] = []
    for part, group in enumerate(groups, start=1):
        parts = len(groups)
        label = unit.label if parts == 1 else f"{unit.label} (part {part} of {parts})"
        text = "\n".join(_texts(group))
        tokens = counter.count(compose_embed_text(unit.heading_path, label, text))
        if tokens > counter.max_tokens:
            raise ChunkingError(
                f"{unit.doc_id}/{unit.locator} part {part}: {tokens} tokens exceeds "
                f"the limit of {counter.max_tokens}"
            )
        blocks: list[Block] = [unit.blocks[i] for i in sorted({i for i, _ in group})]
        chunks.append(
            Chunk(
                id=f"{unit.doc_id}/{unit.locator}" + (f"#{part}" if parts > 1 else ""),
                doc_id=document.doc_id,
                doc_title=document.title,
                doc_kind=document.kind,
                domain=document.domain,
                locator=unit.locator,
                label=label,
                heading_path=unit.heading_path,
                paragraphs=_ordered_unique(b.paragraph for b in blocks),
                pages=_ordered_unique(b.page for b in blocks),
                elements=_ordered_unique(b.element for b in blocks),
                part=part,
                parts=parts,
                text=text,
                token_count=tokens,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return chunks


def chunk_corpus(
    units: Iterable[tuple[SourceDocument, Sequence[SourceUnit]]], counter: TokenCounter
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for document, doc_units in units:
        for unit in doc_units:
            if unit.doc_id != document.doc_id:
                raise ChunkingError(f"unit {unit.locator} belongs to {unit.doc_id}")
            for chunk in chunk_unit(unit, document, counter):
                if chunk.id in seen:
                    raise ChunkingError(f"duplicate chunk ID {chunk.id}")
                seen.add(chunk.id)
                chunks.append(chunk)
    return chunks


# --- manifest ------------------------------------------------------------------


def build_manifest(
    chunks: Sequence[Chunk],
    *,
    chunker_version: int,
    corpus_fingerprint: str,
    tokenizer_fingerprint: str,
    max_tokens: int,
) -> ChunkManifest:
    return ChunkManifest(
        chunker_version=chunker_version,
        corpus_fingerprint=corpus_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        max_tokens=max_tokens,
        chunk_count=len(chunks),
        chunks=tuple(chunks),
    )


def manifest_bytes(manifest: ChunkManifest) -> bytes:
    data = manifest.model_dump(mode="json")
    return (json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_manifest(path: Path, manifest: ChunkManifest) -> str:
    """Write the manifest and return the SHA-256 of the bytes written."""
    data = manifest_bytes(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_manifest(path: Path) -> tuple[ChunkManifest, str]:
    """The manifest and the SHA-256 of its file."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ChunkingError(f"cannot read chunk manifest {path}: {exc}; run make ingest") from exc
    return ChunkManifest.model_validate_json(data), hashlib.sha256(data).hexdigest()
