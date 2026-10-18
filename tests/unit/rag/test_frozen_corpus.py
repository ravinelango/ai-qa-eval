"""Checks on the committed corpus, tokenizer and chunk manifest.

These never skip: the corpus files and the pinned tokenizer are committed, so a
missing or altered file fails here, in offline CI as anywhere else.
"""

from __future__ import annotations

import pytest

from apps.rag.config import RagConfig, file_sha256
from apps.rag.corpus import load_sources, verify_snapshot
from apps.rag.embedding import WordPieceCounter
from apps.rag.ingest import check_manifest, parse_document
from apps.rag.units import SourceUnit

CONFIG = RagConfig.load()
SOURCES = load_sources(CONFIG.resolve(CONFIG.corpus.sources))


def test_every_listed_file_matches_its_checksum() -> None:
    verify_snapshot(SOURCES, CONFIG.resolve(CONFIG.corpus.dir))


@pytest.fixture(scope="module")
def part11() -> dict[str, SourceUnit]:
    document = SOURCES.get("21cfr11")
    data = (CONFIG.resolve(CONFIG.corpus.dir) / document.file).read_bytes()
    return {u.locator: u for u in parse_document(document, data)}


def test_audit_trail_paragraph(part11: dict[str, SourceUnit]) -> None:
    unit = part11["11.10(e)"]
    assert unit.label == "§11.10(e)"
    assert unit.heading_path[-1].startswith("§ 11.10 ")
    assert unit.blocks[0].text.startswith("(e)")
    assert "audit trail" in unit.blocks[0].text.lower()
    assert part11["11.10(i)"].label == "§11.10(i)"  # a letter paragraph, not a roman numeral


def test_signature_manifestations(part11: dict[str, SourceUnit]) -> None:
    assert {"11.50(a)", "11.50(b)"} <= set(part11)
    paragraphs = [b.paragraph for b in part11["11.50(a)"].blocks]
    assert paragraphs == ["§11.50(a)", "§11.50(a)(1)", "§11.50(a)(2)", "§11.50(a)(3)"]


def test_identification_code_and_password_controls(part11: dict[str, SourceUnit]) -> None:
    for letter in "abcde":
        unit = part11[f"11.300({letter})"]
        assert unit.label == f"§11.300({letter})"
        assert unit.heading_path[-1].startswith("§ 11.300 ")


def test_committed_tokenizer_matches_its_pin_and_keeps_its_license() -> None:
    settings = CONFIG.embedding
    path = settings.file_path(settings.tokenizer, CONFIG.root)
    assert path.parent.parts[-2:] == ("models", "bge-small-en-v1.5")
    assert file_sha256(path) == settings.tokenizer.sha256
    assert (path.parent / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    source = (path.parent / "SOURCE.md").read_text(encoding="utf-8")
    assert settings.revision is not None
    assert settings.revision in source
    assert settings.tokenizer.sha256 in source


def test_committed_chunk_manifest_is_reproduced_byte_for_byte() -> None:
    # A missing or mismatched tokenizer raises here: the check fails, never skips.
    counter = WordPieceCounter.from_settings(CONFIG.embedding, CONFIG.root)
    assert check_manifest(
        CONFIG, counter, tokenizer_fingerprint=CONFIG.embedding.tokenizer_fingerprint()
    )
