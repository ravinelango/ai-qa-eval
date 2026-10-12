"""Versioned prompt files and the RAG configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.core.errors import ConfigurationError
from apps.core.evidence import RetrievalRecord, RetrievedChunkRecord
from apps.core.models import ModelRegistry
from apps.rag.config import RagConfig
from apps.rag.prompts import PROMPTS_DIR, load_prompt, render_excerpts


def test_shipped_config_resolves_and_points_at_real_files() -> None:
    config = RagConfig.load()
    assert config.resolve(config.corpus.sources).is_file()
    assert config.resolve(config.generation.prompt).is_file()
    assert config.embedding.max_tokens == 512
    assert config.retrieval.top_k == 5
    # The generator alias exists in the model registry.
    ModelRegistry.load().resolve(config.generation.model)


def test_prompt_version_is_tied_to_file_name_and_content(tmp_path: Path) -> None:
    prompt = load_prompt(PROMPTS_DIR / "answer-v1.yaml")
    assert prompt.tag == "answer@v1"
    assert len(prompt.sha256) == 64
    edited = tmp_path / "answer-v1.yaml"
    edited.write_bytes((PROMPTS_DIR / "answer-v1.yaml").read_bytes() + b"\n# edit\n")
    assert load_prompt(edited).sha256 != prompt.sha256
    misnamed = tmp_path / "answer-v2.yaml"
    misnamed.write_bytes((PROMPTS_DIR / "answer-v1.yaml").read_bytes())
    with pytest.raises(ConfigurationError, match=r"should be named answer-v1\.yaml"):
        load_prompt(misnamed)


def test_excerpts_are_rendered_from_the_saved_retrieval_record() -> None:
    record = RetrievalRecord(
        query="q",
        top_k=1,
        embedder="e",
        embedding_fingerprint="f",
        chunk_manifest_sha256="m",
        results=[
            RetrievedChunkRecord(
                rank=1,
                chunk_id='doc/a"b',
                doc_id="doc",
                doc_kind="guidance",
                label="Section I",
                heading_path=["Title", "I. INTRO"],
                score=0.5,
                text="Body text.",
                text_sha256="0" * 64,
            )
        ],
    )
    assert render_excerpts(record) == (
        '<excerpt id="doc/a&quot;b" kind="guidance" label="Section I">\n'
        "Title > I. INTRO\nBody text.\n</excerpt>"
    )
    prompt = load_prompt(PROMPTS_DIR / "answer-v1.yaml")
    user = prompt.render_user(question="What?", retrieval=record)
    assert user.startswith("Source excerpts:\n\n<excerpt")
    assert user.endswith("Question: What?")
