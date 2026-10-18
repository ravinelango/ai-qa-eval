"""Explicit refresh of the frozen corpus from its official sources.

    make corpus-refresh                     (all documents)
    uv run python -m apps.rag.corpus_fetch --doc 21cfr11
    uv run python -m apps.rag.corpus_fetch --render   (re-render SOURCES.md only)

Ingestion never downloads anything. This command is the only code that does: it
fetches each listed document and checks it is the expected kind of file. Only when
every requested document has been fetched and checked does it write the files into
data/corpus/ and rewrite the snapshot fields in sources.yaml (retrieval date,
SHA-256, size, and for eCFR the as-of date and latest amendment); a failure changes
nothing. Review the diff before committing: a changed SHA-256 means the official
text changed.

Descriptive fields (title, kind, domain, issue date, reuse notice) are written by a
person who has read the document and its reuse notice; this tool never infers them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as SafeET

from apps.core.redact import configure_logging
from apps.rag.config import RagConfig
from apps.rag.corpus import (
    CorpusManifest,
    SourceDocument,
    dump_sources,
    load_sources,
    render_sources_md,
)
from apps.rag.download import Fetcher, https_fetch
from apps.rag.errors import CorpusError, RagError

ECFR_API = "https://www.ecfr.gov/api/versioner/v1"


def _json(fetch: Fetcher, url: str) -> Any:
    try:
        return json.loads(fetch(url))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{url} did not return JSON: {exc}") from exc


def ecfr_up_to_date_as_of(title: int, fetch: Fetcher) -> date:
    """The latest date for which the eCFR has this title's text."""
    data = _json(fetch, f"{ECFR_API}/titles.json")
    for entry in data.get("titles", []):
        if entry.get("number") == title:
            return date.fromisoformat(entry["up_to_date_as_of"])
    raise CorpusError(f"eCFR titles.json has no title {title}")


def ecfr_latest_amendment(title: int, part: str, as_of: date, fetch: Fetcher) -> date:
    data = _json(fetch, f"{ECFR_API}/versions/title-{title}.json?part={part}")
    dates = [
        date.fromisoformat(v["amendment_date"])
        for v in data.get("content_versions", [])
        if v.get("part") == part and v.get("amendment_date")
    ]
    dates = [d for d in dates if d <= as_of]
    if not dates:
        raise CorpusError(f"eCFR lists no versions of title {title} part {part} up to {as_of}")
    return max(dates)


def ecfr_part_url(title: int, part: str, as_of: date) -> str:
    return f"{ECFR_API}/full/{as_of.isoformat()}/title-{title}.xml?part={part}"


def check_ecfr_xml(data: bytes, part: str) -> None:
    try:
        root = SafeET.fromstring(data)
    except SafeET.ParseError as exc:
        raise CorpusError(f"eCFR response is not well-formed XML: {exc}") from exc
    parts = [e for e in root.iter() if e.get("TYPE") == "PART" and e.get("N") == part]
    if len(parts) != 1:
        raise CorpusError(f"eCFR XML should contain exactly one PART {part}, found {len(parts)}")


def check_pdf(data: bytes, url: str) -> None:
    if not data.startswith(b"%PDF-"):
        raise CorpusError(f"{url} did not return a PDF (first bytes {data[:16]!r})")


@dataclass(frozen=True)
class RefreshResult:
    doc_id: str
    old_sha256: str | None
    new_sha256: str

    @property
    def changed(self) -> bool:
        return self.old_sha256 != self.new_sha256


def refresh_document(
    document: SourceDocument, *, fetch: Fetcher, today: date
) -> tuple[SourceDocument, bytes]:
    """Fetch and check one document. Writes nothing: returns the updated entry and bytes."""
    update: dict[str, Any] = {}
    if document.ecfr is not None:
        ref = document.ecfr
        as_of = ecfr_up_to_date_as_of(ref.title, fetch)
        url = ecfr_part_url(ref.title, ref.part, as_of)
        data = fetch(url)
        check_ecfr_xml(data, ref.part)
        latest = ecfr_latest_amendment(ref.title, ref.part, as_of, fetch)
        update["ecfr"] = {**ref.model_dump(), "as_of": as_of, "latest_amendment": latest}
        update["source_url"] = url
    elif document.source_url is not None:  # SourceDocument requires it for guidance
        data = fetch(document.source_url)
        check_pdf(data, document.source_url)
    else:
        raise CorpusError(f"{document.doc_id}: no source_url to fetch")
    update.update(retrieved=today, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return SourceDocument.model_validate({**document.model_dump(), **update}), data


def refresh(
    config: RagConfig,
    *,
    only: Iterable[str] = (),
    fetch: Fetcher = https_fetch,
    today: date | None = None,
) -> list[RefreshResult]:
    sources_path = config.resolve(config.corpus.sources)
    corpus_dir = config.resolve(config.corpus.dir)
    manifest = load_sources(sources_path)
    wanted = set(only)
    unknown = wanted - {d.doc_id for d in manifest.documents}
    if unknown:
        raise CorpusError(f"unknown doc_id(s): {', '.join(sorted(unknown))}")
    day = today or datetime.now(UTC).date()
    results: list[RefreshResult] = []
    documents: list[SourceDocument] = []
    downloads: list[tuple[Path, bytes]] = []
    # Fetch and check everything first; any failure leaves the snapshot untouched.
    for document in manifest.documents:
        if wanted and document.doc_id not in wanted:
            documents.append(document)
            continue
        fresh, data = refresh_document(document, fetch=fetch, today=day)
        results.append(RefreshResult(document.doc_id, document.sha256, fresh.sha256 or ""))
        documents.append(fresh)
        downloads.append((corpus_dir / document.file, data))
    updated = CorpusManifest(schema_version=manifest.schema_version, documents=tuple(documents))
    for path, data in downloads:
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    dump_sources(updated, sources_path)
    (corpus_dir / "SOURCES.md").write_text(render_sources_md(updated), encoding="utf-8")
    return results


def render_only(config: RagConfig) -> Path:
    corpus_dir = config.resolve(config.corpus.dir)
    manifest = load_sources(config.resolve(config.corpus.sources))
    target = corpus_dir / "SOURCES.md"
    target.write_text(render_sources_md(manifest), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--doc", action="append", default=[], help="refresh only this doc_id")
    parser.add_argument("--render", action="store_true", help="only re-render SOURCES.md")
    args = parser.parse_args(argv)
    config = RagConfig.load()
    if args.render:
        print(f"wrote {render_only(config)}")
        return 0
    try:
        results = refresh(config, only=args.doc)
    except (RagError, OSError) as exc:  # OSError covers network failures (URLError)
        print(f"corpus refresh failed: {exc}", file=sys.stderr)
        return 1
    if not results:
        print("sources.yaml lists no documents to refresh; add their entries first")
        return 1
    for r in results:
        status = "CHANGED" if r.changed else "unchanged"
        print(f"{r.doc_id}: {status} sha256={r.new_sha256}")
    print("Review `git diff data/corpus` before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
