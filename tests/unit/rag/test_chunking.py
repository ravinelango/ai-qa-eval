"""Chunking: every chunk fits the token limit with its headings, nothing is lost or
truncated, IDs are stable and unique, and the manifest is byte-for-byte reproducible."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from apps.rag.chunking import (
    Chunk,
    ChunkManifest,
    build_manifest,
    chunk_corpus,
    chunk_unit,
    manifest_bytes,
    read_manifest,
    write_manifest,
)
from apps.rag.corpus import SourceDocument
from apps.rag.embedding import HashingEmbedder, WordPieceCounter, load_tokenizer
from apps.rag.errors import ChunkingError
from apps.rag.parse_ecfr import parse_ecfr_part
from apps.rag.units import Block, SourceUnit
from tests.rag_helpers import ECFR_DOC, SYNTHETIC_ECFR, frozen_entry, make_tokenizer

DOC = SourceDocument.model_validate(frozen_entry(ECFR_DOC, SYNTHETIC_ECFR.read_bytes()))
HEADINGS = ("PART 99—SYNTHETIC", "§ 99.10 Controls.")


def _unit(*texts: str, locator: str = "99.10(a)") -> SourceUnit:
    return SourceUnit(
        doc_id=DOC.doc_id,
        locator=locator,
        label=f"§{locator}",
        heading_path=HEADINGS,
        blocks=tuple(
            Block(text=t, paragraph=f"§99.10(a)({i})" if i else "§99.10(a)", element=f"P[{i}]")
            for i, t in enumerate(texts)
        ),
    )


def _words(text: str) -> list[str]:
    return text.split()


def _check(chunks: list[Chunk], counter: HashingEmbedder) -> None:
    for chunk in chunks:
        assert chunk.token_count == counter.count(chunk.embed_text)
        assert chunk.token_count <= counter.max_tokens
        assert chunk.embed_text.startswith("\n".join(HEADINGS) + "\n" + chunk.label + "\n")


def test_a_unit_that_fits_is_one_chunk() -> None:
    counter = HashingEmbedder(max_tokens=64)
    [chunk] = chunk_unit(_unit("(a) Short text.", "(1) More text."), DOC, counter)
    assert chunk.id == "synthetic-part99/99.10(a)"
    assert (chunk.part, chunk.parts) == (1, 1)
    assert chunk.label == "§99.10(a)"
    assert chunk.text == "(a) Short text.\n(1) More text."
    assert chunk.paragraphs == ("§99.10(a)", "§99.10(a)(1)")
    assert chunk.elements == ("P[0]", "P[1]")
    assert (chunk.doc_kind, chunk.domain) == ("regulation", DOC.domain)
    _check([chunk], counter)


def test_long_units_split_between_blocks_and_keep_every_word() -> None:
    counter = HashingEmbedder(max_tokens=40)
    blocks = [f"({i}) " + " ".join(f"word{i}x{j}" for j in range(12)) for i in range(1, 7)]
    chunks = chunk_unit(_unit(*blocks), DOC, counter)
    assert len(chunks) > 1
    assert [c.id for c in chunks] == [
        f"synthetic-part99/99.10(a)#{k}" for k in range(1, len(chunks) + 1)
    ]
    assert chunks[0].label == f"§99.10(a) (part 1 of {len(chunks)})"
    assert _words(" ".join(c.text for c in chunks)) == _words(" ".join(blocks))
    _check(chunks, counter)
    # Each part lists only the paragraphs it contains.
    assert sum(len(c.paragraphs) for c in chunks) == len(blocks)


def test_oversized_block_splits_by_sentence_then_word() -> None:
    counter = HashingEmbedder(max_tokens=30)
    sentence = "Alpha " + " ".join(f"w{j}" for j in range(60)) + "."
    text = "First sentence is short. " + sentence + " Last one."
    chunks = chunk_unit(_unit(text), DOC, counter)
    assert len(chunks) >= 3
    assert _words(" ".join(c.text for c in chunks)) == _words(text)
    _check(chunks, counter)


def test_a_word_that_cannot_fit_with_its_headings_is_an_error() -> None:
    counter = HashingEmbedder(max_tokens=12)  # the headings alone use most of it
    with pytest.raises(ChunkingError, match="single word does not fit"):
        chunk_unit(_unit(" ".join(["x"] * 40)), DOC, counter)


def test_duplicate_chunk_ids_are_an_error() -> None:
    counter = HashingEmbedder()
    unit = _unit("(a) text")
    with pytest.raises(ChunkingError, match="duplicate chunk ID"):
        chunk_corpus([(DOC, [unit, unit])], counter)


def test_manifest_is_byte_identical_for_identical_inputs(tmp_path: Path) -> None:
    def build() -> bytes:
        units = parse_ecfr_part(SYNTHETIC_ECFR.read_bytes(), doc_id=DOC.doc_id, part="99")
        chunks = chunk_corpus([(DOC, units)], HashingEmbedder(max_tokens=40))
        manifest = build_manifest(
            chunks,
            chunker_version=1,
            corpus_fingerprint="c" * 64,
            tokenizer_fingerprint="t" * 64,
            max_tokens=40,
        )
        return manifest_bytes(manifest)

    first, second = build(), build()
    assert first == second
    sha = write_manifest(tmp_path / "m.json", ChunkManifest.model_validate_json(first))
    manifest, read_sha = read_manifest(tmp_path / "m.json")
    assert sha == read_sha
    assert (tmp_path / "m.json").read_bytes() == first
    assert manifest.chunk_count == len(manifest.chunks) > 30
    assert len({c.id for c in manifest.chunks}) == manifest.chunk_count
    assert any(c.parts > 1 for c in manifest.chunks), "the limit should force some splits"


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ChunkingError, match="run make ingest"):
        read_manifest(tmp_path / "absent.json")


def test_wordpiece_counter_counts_special_tokens_and_never_truncates(tmp_path: Path) -> None:
    path = make_tokenizer(tmp_path / "tokenizer.json", truncate_at=8)
    counter = WordPieceCounter(load_tokenizer(path), max_tokens=512)
    assert counter.count("audit trail") == 4  # [CLS] audit trail [SEP]
    long_text = " ".join(["record"] * 50)
    assert counter.count(long_text) == 52  # truncation in tokenizer.json is ignored
    assert re.fullmatch(r"\d+", str(counter.max_tokens))
