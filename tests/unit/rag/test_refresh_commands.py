"""The explicit refresh commands, driven by a scripted fetcher (no network)."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from apps.rag import model_fetch
from apps.rag.config import EmbeddingSettings, RagConfig
from apps.rag.corpus import load_sources
from apps.rag.corpus_fetch import ECFR_API, refresh
from apps.rag.download import decode_body, https_fetch
from apps.rag.errors import CorpusError, EmbedderError, RagError
from tests.rag_helpers import ECFR_DOC, GUIDANCE_DOC, SYNTHETIC_ECFR, today

PDF = b"%PDF-1.7 synthetic"


class ScriptedFetcher:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        if url not in self.responses:
            raise AssertionError(f"unexpected fetch {url}")
        return self.responses[url]


def _corpus(root: Path, *entries: dict[str, Any]) -> RagConfig:
    corpus = root / "data" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "sources.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "documents": list(entries)}), encoding="utf-8"
    )
    return RagConfig.load(root=root)


def _ecfr_responses(xml: bytes) -> dict[str, bytes]:
    return {
        f"{ECFR_API}/titles.json": json.dumps(
            {"titles": [{"number": 1, "up_to_date_as_of": "2026-01-02"}]}
        ).encode(),
        f"{ECFR_API}/versions/title-1.json?part=99": json.dumps(
            {
                "content_versions": [
                    {"part": "99", "amendment_date": "2019-03-01"},
                    {"part": "99", "amendment_date": "2025-06-01"},
                    {"part": "99", "amendment_date": "2026-05-01"},  # after as_of: ignored
                ]
            }
        ).encode(),
        f"{ECFR_API}/full/2026-01-02/title-1.xml?part=99": xml,
    }


def test_refresh_freezes_ecfr_and_pdf_sources(tmp_path: Path) -> None:
    config = _corpus(tmp_path, ECFR_DOC, GUIDANCE_DOC)
    xml = SYNTHETIC_ECFR.read_bytes()
    fetch = ScriptedFetcher({**_ecfr_responses(xml), GUIDANCE_DOC["source_url"]: PDF})
    results = refresh(config, fetch=fetch, today=today())

    assert [(r.doc_id, r.changed) for r in results] == [
        ("synthetic-part99", True),
        ("synthetic-guidance", True),
    ]
    manifest = load_sources(config.resolve(config.corpus.sources))
    reg, guidance = manifest.documents
    assert reg.is_frozen
    assert guidance.is_frozen
    assert reg.ecfr is not None
    assert (str(reg.ecfr.as_of), str(reg.ecfr.latest_amendment)) == ("2026-01-02", "2025-06-01")
    assert reg.source_url == f"{ECFR_API}/full/2026-01-02/title-1.xml?part=99"
    assert reg.sha256 == hashlib.sha256(xml).hexdigest()
    assert (tmp_path / "data" / "corpus" / "part99.xml").read_bytes() == xml
    assert guidance.bytes == len(PDF)
    assert str(guidance.retrieved) == "2026-01-05"
    sources_md = (tmp_path / "data" / "corpus" / "SOURCES.md").read_text(encoding="utf-8")
    assert reg.sha256 in sources_md

    # Refreshing again with identical bytes reports no change.
    again = refresh(config, only=["synthetic-guidance"], fetch=fetch, today=today())
    assert [(r.doc_id, r.changed) for r in again] == [("synthetic-guidance", False)]


def test_refresh_rejects_unexpected_content(tmp_path: Path) -> None:
    config = _corpus(tmp_path, ECFR_DOC, GUIDANCE_DOC)
    html = b"<html>blocked</html>"
    with pytest.raises(CorpusError, match="did not return a PDF"):
        refresh(
            config,
            only=["synthetic-guidance"],
            fetch=ScriptedFetcher({GUIDANCE_DOC["source_url"]: html}),
        )
    wrong_part = b'<DIV5 N="11" TYPE="PART"><HEAD>x</HEAD></DIV5>'
    with pytest.raises(CorpusError, match="exactly one PART 99"):
        refresh(
            config, only=["synthetic-part99"], fetch=ScriptedFetcher(_ecfr_responses(wrong_part))
        )
    with pytest.raises(CorpusError, match="unknown doc_id"):
        refresh(config, only=["nope"], fetch=ScriptedFetcher({}))
    # Nothing was written for the failed documents.
    assert not (tmp_path / "data" / "corpus" / "guidance.pdf").exists()


def test_a_failed_refresh_changes_nothing(tmp_path: Path) -> None:
    config = _corpus(tmp_path, ECFR_DOC, GUIDANCE_DOC)
    xml = SYNTHETIC_ECFR.read_bytes()
    refresh(
        config, fetch=ScriptedFetcher({**_ecfr_responses(xml), GUIDANCE_DOC["source_url"]: PDF})
    )
    corpus = tmp_path / "data" / "corpus"
    before = {p.name: p.read_bytes() for p in corpus.iterdir()}

    class FailsOnPdf(ScriptedFetcher):
        def __call__(self, url: str) -> bytes:
            if url == GUIDANCE_DOC["source_url"]:
                raise OSError("network down")
            return super().__call__(url)

    changed_xml = xml.replace(b"Placeholder scope sentence", b"Revised scope sentence")
    with pytest.raises(OSError, match="network down"):
        refresh(config, fetch=FailsOnPdf(_ecfr_responses(changed_xml)))
    assert {p.name: p.read_bytes() for p in corpus.iterdir()} == before


def test_downloads_are_https_only() -> None:
    with pytest.raises(RagError, match="non-HTTPS"):
        https_fetch("http://www.fda.gov/media/1/download")


def test_compressed_responses_are_decoded() -> None:
    assert decode_body(gzip.compress(b"<xml/>"), "gzip") == b"<xml/>"
    assert decode_body(b"%PDF-1.7", None) == b"%PDF-1.7"
    with pytest.raises(RagError, match="unsupported Content-Encoding"):
        decode_body(b"x", "br")


def _settings(tmp_path: Path, **pins: str | None) -> EmbeddingSettings:
    data: dict[str, Any] = RagConfig.load().embedding.model_dump(mode="json")
    data["local_dir"] = str(tmp_path / "models")
    data["revision"] = pins.get("revision")
    data["model"]["sha256"] = pins.get("model")
    data["tokenizer"]["sha256"] = pins.get("tokenizer")
    return EmbeddingSettings.model_validate(data)


REV = "a" * 40
MODEL, TOKENIZER = b"onnx bytes", b"tokenizer bytes"


def _hf(revision: str) -> dict[str, bytes]:
    base = f"https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/{revision}"
    return {f"{base}/onnx/model.onnx": MODEL, f"{base}/tokenizer.json": TOKENIZER}


def test_model_fetch_installs_only_files_matching_their_pins(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        revision=REV,
        model=hashlib.sha256(MODEL).hexdigest(),
        tokenizer=hashlib.sha256(TOKENIZER).hexdigest(),
    )
    files = model_fetch.fetch_pinned(settings, tmp_path, ScriptedFetcher(_hf(REV)))
    assert [f.path for f in files] == ["onnx/model.onnx", "tokenizer.json"]
    assert (tmp_path / "models" / "onnx" / "model.onnx").read_bytes() == MODEL

    tampered = {**_hf(REV), next(iter(_hf(REV))): b"other bytes"}
    (tmp_path / "models" / "onnx" / "model.onnx").unlink()
    with pytest.raises(EmbedderError, match="does not match the pin"):
        model_fetch.fetch_pinned(settings, tmp_path, ScriptedFetcher(tampered))
    assert not (tmp_path / "models" / "onnx" / "model.onnx").exists()


def test_model_fetch_requires_pins_and_pin_mode_changes_nothing(tmp_path: Path) -> None:
    unpinned = _settings(tmp_path)
    with pytest.raises(EmbedderError, match="must be set"):
        model_fetch.fetch_pinned(unpinned, tmp_path, ScriptedFetcher({}))
    api = {
        "https://huggingface.co/api/models/BAAI/bge-small-en-v1.5": json.dumps(
            {"sha": REV}
        ).encode()
    }
    revision, files = model_fetch.propose_pins(unpinned, ScriptedFetcher({**api, **_hf(REV)}))
    assert revision == REV
    assert {f.path: f.sha256 for f in files} == {
        "onnx/model.onnx": hashlib.sha256(MODEL).hexdigest(),
        "tokenizer.json": hashlib.sha256(TOKENIZER).hexdigest(),
    }
    assert not (tmp_path / "models").exists()
