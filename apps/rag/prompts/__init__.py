"""Versioned prompt files. Each file is `<id>-<version>.yaml`; see answer-v1.yaml."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path
from string import Template

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apps.core.errors import ConfigurationError
from apps.core.evidence import RetrievalRecord

PROMPTS_DIR = Path(__file__).resolve().parent


class PromptTemplate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(pattern=r"^v\d+$")
    description: str
    system: str
    user: str
    # SHA-256 of the file's bytes; set by load_prompt.
    sha256: str = ""

    @property
    def tag(self) -> str:
        return f"{self.id}@{self.version}"

    def render_user(self, *, question: str, retrieval: RetrievalRecord) -> str:
        return Template(self.user).substitute(
            excerpts=render_excerpts(retrieval), question=question
        )


def load_prompt(path: Path) -> PromptTemplate:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ConfigurationError(f"cannot read prompt {path}: {exc}") from exc
    try:
        prompt = PromptTemplate.model_validate(
            {**yaml.safe_load(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    except ValidationError as exc:
        raise ConfigurationError(f"invalid prompt file {path}:\n{exc}") from exc
    if path.name != f"{prompt.id}-{prompt.version}.yaml":
        raise ConfigurationError(f"{path.name} should be named {prompt.id}-{prompt.version}.yaml")
    return prompt


def render_excerpts(retrieval: RetrievalRecord) -> str:
    """The retrieved chunks as tagged excerpts, built only from the saved record."""
    parts: list[str] = []
    for r in retrieval.results:
        attrs = " ".join(
            f'{name}="{html.escape(value, quote=True)}"'
            for name, value in (("id", r.chunk_id), ("kind", r.doc_kind), ("label", r.label))
        )
        context = " > ".join(r.heading_path)
        parts.append(f"<excerpt {attrs}>\n{context}\n{r.text}\n</excerpt>")
    return "\n\n".join(parts)
