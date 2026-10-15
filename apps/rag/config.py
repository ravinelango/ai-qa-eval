"""Loads config/rag.yaml and computes the fingerprints that tie an index to its inputs.

Fingerprints are SHA-256 over canonical JSON, so key order and formatting in the
YAML never change them; only values do.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apps.core.errors import ConfigurationError
from apps.core.models import REPO_ROOT

DEFAULT_RAG_CONFIG = REPO_ROOT / "config" / "rag.yaml"

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def canonical_sha256(data: Any) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class CorpusSettings(_Strict):
    sources: Path
    dir: Path


class ChunkingSettings(_Strict):
    version: int = Field(ge=1)
    manifest: Path


class PinnedFile(_Strict):
    path: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    # Where this file is kept, when not under embedding.local_dir (e.g. a committed copy).
    local_dir: Path | None = None


class PinnedModelSettings(_Strict):
    """A local ONNX model taken from one Hugging Face repo revision, its files pinned."""

    # The config/rag.yaml section, for messages.
    section: ClassVar[str] = "model"

    name: str
    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    model: PinnedFile
    tokenizer: PinnedFile
    local_dir: Path
    max_tokens: int = Field(gt=2)
    threads: int = Field(ge=1)
    batch_size: int = Field(ge=1)

    @property
    def is_pinned(self) -> bool:
        return (
            self.revision is not None
            and self.model.sha256 is not None
            and self.tokenizer.sha256 is not None
        )

    def file_path(self, pin: PinnedFile, root: Path) -> Path:
        """Where a pinned file is kept on disk."""
        directory = pin.local_dir or self.local_dir
        return (directory if directory.is_absolute() else root / directory) / pin.path

    def fingerprint(self) -> str:
        """Everything that determines the outputs, except where the files are stored."""
        exclude: dict[str, Any] = {
            "local_dir": True,
            "batch_size": True,
            "model": {"local_dir"},
            "tokenizer": {"local_dir"},
        }
        return canonical_sha256(self.model_dump(mode="json", exclude=exclude))


class EmbeddingSettings(PinnedModelSettings):
    section: ClassVar[str] = "embedding"

    dimensions: int = Field(gt=0)
    pooling: Literal["cls"]
    normalize: bool
    query_instruction: str

    def tokenizer_fingerprint(self) -> str:
        """What determines token counts, and therefore chunk boundaries."""
        return canonical_sha256(
            {
                "repo": self.repo,
                "revision": self.revision,
                "tokenizer": self.tokenizer.model_dump(mode="json", exclude={"local_dir"}),
                "max_tokens": self.max_tokens,
            }
        )


class RerankerSettings(PinnedModelSettings):
    """A cross-encoder that scores (question, passage) pairs; used by hybrid_rerank."""

    section: ClassVar[str] = "reranker"


class IndexSettings(_Strict):
    path: Path
    collection: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")
    space: Literal["cosine"]
    ef_construction: int = Field(gt=0)
    ef_search: int = Field(gt=0)


class Bm25Settings(_Strict):
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)


class RetrievalSettings(_Strict):
    # Bump when a code change alters the ranking (keyword analysis, citation rules,
    # fusion), so records from before and after cannot pass as the same method.
    version: int = Field(ge=1)
    top_k: int = Field(ge=1, le=50)
    mode: Literal["dense", "hybrid", "hybrid_rerank"]
    # Hybrid modes: how many chunks each ranking (embedding, keyword) contributes
    # to the fusion, and how many of the fused chunks the reranker scores.
    candidates: int = Field(ge=1)
    bm25: Bm25Settings
    # Reciprocal rank fusion constant: a chunk scores 1 / (rrf_k + rank) per ranking.
    rrf_k: int = Field(ge=1)

    def fingerprint(self) -> str:
        """What determines the ranking. top_k is left out; each record states it."""
        return canonical_sha256(self.model_dump(mode="json", exclude={"top_k"}))


class GenerationSettings(_Strict):
    model: str
    max_tokens: int = Field(gt=0)
    prompt: Path


class RagConfig(_Strict):
    corpus: CorpusSettings
    chunking: ChunkingSettings
    embedding: EmbeddingSettings
    reranker: RerankerSettings
    index: IndexSettings
    retrieval: RetrievalSettings
    generation: GenerationSettings
    # Relative paths in the file resolve against this directory.
    root: Path = REPO_ROOT

    @classmethod
    def load(cls, path: Path = DEFAULT_RAG_CONFIG, *, root: Path = REPO_ROOT) -> RagConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigurationError(f"cannot read RAG config {path}: {exc}") from exc
        try:
            return cls.model_validate({**raw, "root": root})
        except ValidationError as exc:
            raise ConfigurationError(f"invalid RAG config {path}:\n{exc}") from exc

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path
