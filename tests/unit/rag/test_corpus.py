"""The corpus manifest: required provenance, frozen-snapshot checks, fingerprints,
and SOURCES.md kept in step with sources.yaml."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from apps.rag.config import RagConfig
from apps.rag.corpus import (
    CorpusManifest,
    SourceDocument,
    dump_sources,
    load_sources,
    render_sources_md,
    verify_snapshot,
)
from apps.rag.errors import CorpusError
from tests.rag_helpers import ECFR_DOC, GUIDANCE_DOC, frozen_entry

DATA = b"<DIV5 N='99' TYPE='PART'/>"


def _manifest(*entries: dict[str, Any]) -> CorpusManifest:
    return CorpusManifest.model_validate({"schema_version": 1, "documents": list(entries)})


def test_shipped_manifest_is_valid_and_sources_md_is_its_rendering() -> None:
    config = RagConfig.load()
    manifest = load_sources(config.resolve(config.corpus.sources))
    rendered = render_sources_md(manifest)
    on_disk = (config.resolve(config.corpus.dir) / "SOURCES.md").read_text(encoding="utf-8")
    assert on_disk == rendered


def test_shipped_corpus_state_is_reported_not_skipped() -> None:
    config = RagConfig.load()
    manifest = load_sources(config.resolve(config.corpus.sources))
    if manifest.documents:
        verify_snapshot(manifest, config.resolve(config.corpus.dir))
    else:
        with pytest.raises(CorpusError, match="lists no documents"):
            verify_snapshot(manifest, config.resolve(config.corpus.dir))


def test_kind_format_and_status_rules() -> None:
    with pytest.raises(ValidationError, match="Input should be 'final'"):
        SourceDocument.model_validate({**GUIDANCE_DOC, "status": "draft"})
    with pytest.raises(ValidationError, match="taken from eCFR XML"):
        SourceDocument.model_validate({**ECFR_DOC, "ecfr": None})
    with pytest.raises(ValidationError, match="official PDF"):
        SourceDocument.model_validate({**GUIDANCE_DOC, "format": "ecfr-xml"})
    with pytest.raises(ValidationError, match="issue date"):
        SourceDocument.model_validate({**GUIDANCE_DOC, "version_date": None})
    with pytest.raises(ValidationError, match="https"):
        SourceDocument.model_validate({**GUIDANCE_DOC, "source_url": "http://x.gov/a.pdf"})
    with pytest.raises(ValidationError, match="duplicate doc_id"):
        _manifest(GUIDANCE_DOC, {**GUIDANCE_DOC, "file": "other.pdf"})


def test_frozen_means_every_snapshot_field_is_set() -> None:
    assert not SourceDocument.model_validate(GUIDANCE_DOC).is_frozen
    assert not SourceDocument.model_validate(ECFR_DOC).is_frozen
    assert SourceDocument.model_validate(frozen_entry(ECFR_DOC, DATA)).is_frozen
    no_as_of = frozen_entry(ECFR_DOC, DATA)
    no_as_of["ecfr"] = {**no_as_of["ecfr"], "as_of": None}
    assert not SourceDocument.model_validate(no_as_of).is_frozen


def test_verify_snapshot(tmp_path: Path) -> None:
    (tmp_path / "part99.xml").write_bytes(DATA)
    verify_snapshot(_manifest(frozen_entry(ECFR_DOC, DATA)), tmp_path)

    with pytest.raises(CorpusError, match="has SHA-256"):
        verify_snapshot(_manifest(frozen_entry(ECFR_DOC, DATA + b" ")), tmp_path)
    with pytest.raises(CorpusError, match="not downloaded yet"):
        verify_snapshot(_manifest(ECFR_DOC), tmp_path)
    (tmp_path / "part99.xml").unlink()
    with pytest.raises(CorpusError, match="is missing"):
        verify_snapshot(_manifest(frozen_entry(ECFR_DOC, DATA)), tmp_path)


def test_fingerprint_follows_content_and_identity_not_fetch_details() -> None:
    base = frozen_entry(ECFR_DOC, DATA)
    fingerprint = _manifest(base).fingerprint()
    refetched = {**base, "retrieved": "2027-01-01", "source_url": "https://example.invalid/b"}
    assert _manifest(refetched).fingerprint() == fingerprint
    changed = {**base, "sha256": hashlib.sha256(b"new text").hexdigest()}
    assert _manifest(changed).fingerprint() != fingerprint
    redomained = {**base, "domain": "another domain"}
    assert _manifest(redomained).fingerprint() != fingerprint


def test_dump_and_load_round_trip(tmp_path: Path) -> None:
    manifest = _manifest(frozen_entry(ECFR_DOC, DATA), GUIDANCE_DOC)
    path = tmp_path / "sources.yaml"
    dump_sources(manifest, path)
    assert path.read_text(encoding="utf-8").startswith("# Corpus manifest")
    assert load_sources(path) == manifest


def test_sources_md_lists_provenance_for_each_document() -> None:
    text = render_sources_md(_manifest(frozen_entry(ECFR_DOC, DATA), GUIDANCE_DOC))
    assert "| `synthetic-part99` |" in text
    assert "- **Kind:** regulation (final)" in text
    assert "- **Applicable domain:** Synthetic ledgers (test only)" in text
    assert "as of 2026-01-02; latest amendment 2025-06-01" in text
    assert hashlib.sha256(DATA).hexdigest() in text
    assert "- **Retrieved:** not yet" in text  # the unfrozen guidance entry
