"""The real reranker refuses unpinned or mismatched files, and the default suite cannot
load it."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from apps.rag.config import RagConfig, RerankerSettings
from apps.rag.errors import EmbedderError
from apps.rag.rerank import OnnxReranker
from tests.rag_helpers import make_tokenizer


def _pinned(tmp_path: Path, **files: bytes) -> RerankerSettings:
    data: dict[str, Any] = RagConfig.load().reranker.model_dump(mode="json")
    data["local_dir"] = str(tmp_path)
    for which, content in files.items():
        data[which]["sha256"] = hashlib.sha256(content).hexdigest()
    return RerankerSettings.model_validate(data)


def test_shipped_reranker_is_pinned() -> None:
    settings = RagConfig.load().reranker
    assert settings.is_pinned
    assert settings.max_tokens == 512
    assert settings.threads == 1


def test_unpinned_or_missing_files_are_refused(tmp_path: Path) -> None:
    unpinned = RerankerSettings.model_validate(
        {**_pinned(tmp_path).model_dump(mode="json"), "revision": None}
    )
    with pytest.raises(EmbedderError, match=r"not pinned: set reranker\.revision"):
        OnnxReranker(unpinned, tmp_path)
    with pytest.raises(EmbedderError, match="is missing; run make model-fetch"):
        OnnxReranker(_pinned(tmp_path, model=b"m", tokenizer=b"t"), tmp_path)


def test_the_flag_stops_the_real_reranker_loading(tmp_path: Path) -> None:
    tokenizer = make_tokenizer(tmp_path / "tokenizer.json").read_bytes()
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"not a real model")
    settings = _pinned(tmp_path, model=b"not a real model", tokenizer=tokenizer)
    with pytest.raises(EmbedderError, match="forbidden"):
        OnnxReranker(settings, tmp_path)


def test_the_reranker_fingerprint_ignores_where_files_are_kept(tmp_path: Path) -> None:
    base = _pinned(tmp_path, model=b"m", tokenizer=b"t")
    moved = RerankerSettings.model_validate(
        {**base.model_dump(mode="json"), "local_dir": "/elsewhere", "batch_size": 3}
    )
    assert moved.fingerprint() == base.fingerprint()
    assert _pinned(tmp_path, model=b"m2", tokenizer=b"t").fingerprint() != base.fingerprint()
