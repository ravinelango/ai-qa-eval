"""Embedders refuse over-limit input, the real model refuses to load unpinned or
mismatched files, and the default suite cannot load it at all."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any

import pytest

from apps.rag.config import EmbeddingSettings, RagConfig
from apps.rag.embedding import (
    FORBID_MODEL_LOAD_ENV,
    HashingEmbedder,
    OnnxEmbedder,
    WordPieceCounter,
    load_onnx_session,
)
from apps.rag.errors import EmbedderError
from tests.rag_helpers import make_tokenizer

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_deterministic_normalized_and_lexical() -> None:
    embedder = HashingEmbedder()
    first = embedder.embed_documents(["keep an audit trail of record changes"])[0]
    again = HashingEmbedder().embed_query("keep an audit trail of record changes")
    assert first == again
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0)
    near = embedder.embed_query("audit trail for record changes")
    far = embedder.embed_query("password aging policy")
    assert _cosine(first, near) > _cosine(first, far)
    assert embedder.embed_query("") == embedder.embed_query("!!")  # no words: fixed vector


def test_hashing_embedder_refuses_input_over_its_limit() -> None:
    embedder = HashingEmbedder(max_tokens=5)
    with pytest.raises(EmbedderError, match="over the 5-token limit"):
        embedder.embed_documents(["one two three four"])  # 4 words + 2 specials


def _pinned(tmp_path: Path, **files: bytes) -> EmbeddingSettings:
    shipped = RagConfig.load().embedding
    data: dict[str, Any] = shipped.model_dump(mode="json")
    data["revision"] = REVISION
    data["local_dir"] = str(tmp_path)
    data["tokenizer"]["local_dir"] = None
    for which, content in files.items():
        data[which]["sha256"] = hashlib.sha256(content).hexdigest()
    return EmbeddingSettings.model_validate(data)


def test_unpinned_model_refuses_to_start(tmp_path: Path) -> None:
    unpinned = EmbeddingSettings.model_validate(
        {**_pinned(tmp_path, model=b"m", tokenizer=b"t").model_dump(mode="json"), "revision": None}
    )
    assert unpinned.is_pinned is False
    with pytest.raises(EmbedderError, match="not pinned"):
        OnnxEmbedder(unpinned, tmp_path)


def test_missing_or_mismatched_model_files_are_refused(tmp_path: Path) -> None:
    settings = _pinned(tmp_path, model=b"model bytes", tokenizer=b"tokenizer bytes")
    with pytest.raises(EmbedderError, match="is missing; run make model-fetch"):
        OnnxEmbedder(settings, tmp_path)
    (tmp_path / "tokenizer.json").write_bytes(b"different bytes")
    with pytest.raises(EmbedderError, match="pinned"):
        OnnxEmbedder(settings, tmp_path)


def test_default_suite_sets_the_flag_before_anything_runs(pytestconfig: pytest.Config) -> None:
    if pytestconfig.getoption("--run-model"):
        pytest.skip("--run-model lifts the flag")
    assert os.environ[FORBID_MODEL_LOAD_ENV] == "1"


def test_the_flag_stops_the_real_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FORBID_MODEL_LOAD_ENV, "1")
    tokenizer = make_tokenizer(tmp_path / "tokenizer.json").read_bytes()
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"not a real model")
    settings = _pinned(tmp_path, model=b"not a real model", tokenizer=tokenizer)
    # Pins and files check out, so construction gets as far as loading the model,
    # where the flag stops it, however the loader is reached.
    with pytest.raises(EmbedderError, match="loading the real embedding model is forbidden"):
        OnnxEmbedder(settings, tmp_path)
    with pytest.raises(EmbedderError, match="forbidden"):
        load_onnx_session(tmp_path / "onnx" / "model.onnx", 1)


def test_tokenizer_alone_can_be_used_for_counting(tmp_path: Path) -> None:
    tokenizer = make_tokenizer(tmp_path / "tokenizer.json").read_bytes()
    settings = _pinned(tmp_path, model=b"unused", tokenizer=tokenizer)
    counter = WordPieceCounter.from_settings(settings, tmp_path)
    assert counter.count("audit trail") == 4
    assert counter.max_tokens == 512


def test_fingerprints_track_what_determines_vectors_and_tokens(tmp_path: Path) -> None:
    base = _pinned(tmp_path, model=b"m", tokenizer=b"t")
    moved = EmbeddingSettings.model_validate(
        {**base.model_dump(mode="json"), "local_dir": "/elsewhere", "batch_size": 99}
    )
    assert moved.fingerprint() == base.fingerprint()
    other_model = _pinned(tmp_path, model=b"m2", tokenizer=b"t")
    assert other_model.fingerprint() != base.fingerprint()
    assert other_model.tokenizer_fingerprint() == base.tokenizer_fingerprint()
    other_instruction = EmbeddingSettings.model_validate(
        {**base.model_dump(mode="json"), "query_instruction": ""}
    )
    assert other_instruction.fingerprint() != base.fingerprint()
