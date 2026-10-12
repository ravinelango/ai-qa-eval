"""The corpus manifest (data/corpus/sources.yaml) and frozen-snapshot verification.

Each source document is recorded with what it is (regulation or guidance, issuer,
applicable domain), which version it is (issue date, or the eCFR "as of" date and
latest amendment), where it came from (official URL, retrieval date) and its
SHA-256. Ingestion reads only these committed files and stops if any is missing or
differs from its checksum. Refreshing from the official sources is a separate,
explicit step (apps.rag.corpus_fetch).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from apps.rag.config import SHA256_PATTERN, canonical_sha256, file_sha256
from apps.rag.errors import CorpusError

SOURCES_SCHEMA_VERSION = 1

SourceKind = Literal["regulation", "guidance"]
SourceFormat = Literal["ecfr-xml", "pdf"]

SOURCES_HEADER = """\
# Corpus manifest (docs/decisions/0006-rag-corpus-and-index.md). SOURCES.md is
# rendered from this file. Ingestion reads only the files listed here and refuses
# any whose SHA-256 differs. `make corpus-refresh` re-downloads from source_url and
# rewrites the snapshot fields (retrieved, sha256, bytes, eCFR dates).
"""


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EcfrReference(_Strict):
    """Where a regulation lives in the eCFR, and which point-in-time text was taken."""

    title: int = Field(ge=1, le=50)
    part: str = Field(pattern=r"^\d+[A-Z]?$")
    # The eCFR date the snapshot reflects ("up to date as of"). Set by refresh.
    as_of: date | None = None
    # Latest amendment to this part on or before `as_of`. Set by refresh.
    latest_amendment: date | None = None


class PdfOptions(_Strict):
    """Per-document parsing options for guidance PDFs."""

    # 1-based pages to leave out entirely (e.g. a table of contents).
    skip_pages: tuple[int, ...] = ()


class SourceDocument(_Strict):
    doc_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    title: str = Field(min_length=1)
    kind: SourceKind
    issuer: str = Field(min_length=1)
    # The products, records or activities the document applies to, in its own terms.
    domain: str = Field(min_length=1)
    # Drafts are excluded from the corpus: guidance must be final, and a regulation's
    # text is the codified text.
    status: Literal["final"]
    # Guidance: the issue date printed on the document. Regulation: see ecfr.
    version_date: date | None = None
    version_label: str = Field(min_length=1)
    format: SourceFormat
    file: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    source_url: str | None = Field(default=None, pattern=r"^https://")
    landing_url: str | None = Field(default=None, pattern=r"^https://")
    reuse_notice: str = Field(min_length=1)
    ecfr: EcfrReference | None = None
    pdf: PdfOptions | None = None
    # Snapshot fields: null until the file has been downloaded.
    retrieved: date | None = None
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    bytes: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _consistent(self) -> SourceDocument:
        if self.kind == "regulation":
            if self.format != "ecfr-xml" or self.ecfr is None:
                raise ValueError(f"{self.doc_id}: a regulation is taken from eCFR XML")
            if self.pdf is not None:
                raise ValueError(f"{self.doc_id}: pdf options apply to guidance only")
        else:
            if self.format != "pdf" or self.ecfr is not None:
                raise ValueError(f"{self.doc_id}: guidance is taken from the official PDF")
            if self.source_url is None:
                raise ValueError(f"{self.doc_id}: guidance needs its official source_url")
            if self.version_date is None:
                raise ValueError(f"{self.doc_id}: guidance needs the issue date it carries")
        return self

    @property
    def is_frozen(self) -> bool:
        snapshot = (self.retrieved, self.sha256, self.bytes, self.source_url)
        if any(value is None for value in snapshot):
            return False
        if self.ecfr is not None:
            return self.ecfr.as_of is not None and self.ecfr.latest_amendment is not None
        return True

    def identity(self) -> dict[str, object]:
        """The fields that flow into chunks and the corpus fingerprint.

        Where and when the file was fetched are left out: an identical file fetched
        on another day yields the same chunks.
        """
        return self.model_dump(
            mode="json",
            exclude={"source_url", "landing_url", "reuse_notice", "retrieved", "bytes"},
        )


class CorpusManifest(_Strict):
    schema_version: Literal[1]
    documents: tuple[SourceDocument, ...]

    @model_validator(mode="after")
    def _unique(self) -> CorpusManifest:
        for field in ("doc_id", "file"):
            values = [getattr(d, field) for d in self.documents]
            duplicates = sorted({v for v in values if values.count(v) > 1})
            if duplicates:
                raise ValueError(f"duplicate {field}: {', '.join(duplicates)}")
        return self

    def fingerprint(self) -> str:
        return canonical_sha256([d.identity() for d in self.documents])

    def get(self, doc_id: str) -> SourceDocument:
        for document in self.documents:
            if document.doc_id == doc_id:
                return document
        raise CorpusError(f"no document {doc_id!r} in the corpus manifest")


def load_sources(path: Path) -> CorpusManifest:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusError(f"cannot read corpus manifest {path}: {exc}") from exc
    try:
        return CorpusManifest.model_validate(raw)
    except ValidationError as exc:
        raise CorpusError(f"invalid corpus manifest {path}:\n{exc}") from exc


def dump_sources(manifest: CorpusManifest, path: Path) -> None:
    data = manifest.model_dump(mode="json", exclude_none=True)
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(SOURCES_HEADER + body, encoding="utf-8")


def verify_snapshot(manifest: CorpusManifest, corpus_dir: Path) -> None:
    """Stop unless every listed file is downloaded, present and matches its checksum."""
    if not manifest.documents:
        raise CorpusError(
            "the corpus manifest lists no documents; nothing has been downloaded yet "
            "(see data/corpus/SOURCES.md)"
        )
    problems: list[str] = []
    for document in manifest.documents:
        path = corpus_dir / document.file
        if not document.is_frozen:
            problems.append(f"{document.doc_id}: not downloaded yet (snapshot fields are null)")
        elif not path.is_file():
            problems.append(f"{document.doc_id}: {path} is missing")
        else:
            actual = file_sha256(path)
            if actual != document.sha256:
                problems.append(
                    f"{document.doc_id}: {path.name} has SHA-256 {actual}, "
                    f"manifest says {document.sha256}"
                )
            elif path.stat().st_size != document.bytes:
                problems.append(f"{document.doc_id}: {path.name} size differs from manifest")
    if problems:
        raise CorpusError(
            "corpus snapshot check failed; ingestion uses only the frozen files:\n  "
            + "\n  ".join(problems)
        )


def render_sources_md(manifest: CorpusManifest) -> str:
    """SOURCES.md, generated from the manifest so the two cannot drift apart."""
    lines = [
        "# Corpus sources",
        "",
        "Generated from `sources.yaml` by `python -m apps.rag.corpus_fetch --render`."
        " Do not edit by hand.",
        "",
    ]
    if not manifest.documents:
        lines += [
            "No documents have been downloaded yet.",
            "",
            "Planned (approved Phase 1a scope): 21 CFR Part 11 from the eCFR, plus about",
            "three final FDA guidance documents on Part 11 scope and application, data",
            "integrity, and computer software assurance. Each will be listed here with its",
            "title, kind (regulation or guidance), applicable domain, version or",
            "publication date, eCFR as-of date for the regulation, official source URL,",
            "retrieval date, SHA-256 and the reuse notice that was checked.",
            "",
        ]
        return "\n".join(lines)
    lines += [
        "| ID | Title | Kind | Version | Retrieved | SHA-256 |",
        "|---|---|---|---|---|---|",
    ]
    for d in manifest.documents:
        lines.append(
            f"| `{d.doc_id}` | {d.title} | {d.kind} | {d.version_label} | "
            f"{d.retrieved or 'not yet'} | `{d.sha256 or 'not yet'}` |"
        )
    lines.append("")
    for d in manifest.documents:
        lines += [
            f"## `{d.doc_id}`: {d.title}",
            "",
            f"- **Kind:** {d.kind} ({d.status}), issued by {d.issuer}",
            f"- **Applicable domain:** {d.domain}",
            f"- **Version:** {d.version_label}"
            + (f" (dated {d.version_date})" if d.version_date else ""),
        ]
        if d.ecfr is not None:
            lines.append(
                f"- **eCFR:** title {d.ecfr.title}, part {d.ecfr.part}; as of "
                f"{d.ecfr.as_of or 'not yet'}; latest amendment "
                f"{d.ecfr.latest_amendment or 'not yet'}"
            )
        lines += [
            f"- **File:** `{d.file}` ({d.format}, {d.bytes or '?'} bytes)",
            f"- **Source URL:** {d.source_url or 'not yet'}",
        ]
        if d.landing_url:
            lines.append(f"- **Landing page:** {d.landing_url}")
        lines += [
            f"- **Retrieved:** {d.retrieved or 'not yet'}",
            f"- **SHA-256:** `{d.sha256 or 'not yet'}`",
            f"- **Reuse notice:** {d.reuse_notice}",
            "",
        ]
    return "\n".join(lines)
