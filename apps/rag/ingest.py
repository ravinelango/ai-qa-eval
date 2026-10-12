"""Build the chunk manifest and the vector index from the frozen corpus.

    make ingest                                     (manifest + embeddings + index)
    uv run python -m apps.rag.ingest --check        (verify the saved manifest only)

Steps: verify every corpus file against its SHA-256 in sources.yaml; parse each
document into structural units; chunk them to fit the embedding model; write the
chunk manifest (data/index/chunk_manifest.json, committed); embed every chunk and
sync the local Chroma index (data/chroma/, not committed).

Nothing is downloaded here. `--check` rebuilds the manifest in memory and compares
it byte for byte with the saved file; it needs only the pinned tokenizer, not the
model.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from apps.core.redact import configure_logging
from apps.rag.chunking import (
    ChunkManifest,
    TokenCounter,
    build_manifest,
    chunk_corpus,
    manifest_bytes,
    write_manifest,
)
from apps.rag.config import RagConfig
from apps.rag.corpus import CorpusManifest, SourceDocument, load_sources, verify_snapshot
from apps.rag.embedding import Embedder, OnnxEmbedder, WordPieceCounter
from apps.rag.errors import RagError, SourceParseError
from apps.rag.index import ChunkIndex, IndexStamp, SyncReport
from apps.rag.parse_ecfr import parse_ecfr_part
from apps.rag.parse_guidance import parse_guidance, read_pdf
from apps.rag.units import SourceUnit


def parse_document(document: SourceDocument, data: bytes) -> list[SourceUnit]:
    if document.format == "ecfr-xml":
        if document.ecfr is None:
            raise SourceParseError(f"{document.doc_id}: no eCFR reference")
        return parse_ecfr_part(data, doc_id=document.doc_id, part=document.ecfr.part)
    return parse_guidance(
        read_pdf(data), doc_id=document.doc_id, title=document.title, options=document.pdf
    )


def build_chunk_manifest(
    config: RagConfig,
    sources: CorpusManifest,
    counter: TokenCounter,
    *,
    tokenizer_fingerprint: str,
) -> ChunkManifest:
    corpus_dir = config.resolve(config.corpus.dir)
    verify_snapshot(sources, corpus_dir)
    units = [
        (doc, parse_document(doc, (corpus_dir / doc.file).read_bytes()))
        for doc in sources.documents
    ]
    return build_manifest(
        chunk_corpus(units, counter),
        chunker_version=config.chunking.version,
        corpus_fingerprint=sources.fingerprint(),
        tokenizer_fingerprint=tokenizer_fingerprint,
        max_tokens=counter.max_tokens,
    )


@dataclass(frozen=True)
class IngestReport:
    chunk_count: int
    manifest_path: Path
    manifest_sha256: str
    manifest_changed: bool
    sync: SyncReport


def ingest(config: RagConfig, embedder: Embedder, *, tokenizer_fingerprint: str) -> IngestReport:
    sources = load_sources(config.resolve(config.corpus.sources))
    manifest = build_chunk_manifest(
        config, sources, embedder, tokenizer_fingerprint=tokenizer_fingerprint
    )
    path = config.resolve(config.chunking.manifest)
    previous = path.read_bytes() if path.is_file() else None
    sha = write_manifest(path, manifest)
    vectors = embedder.embed_documents([c.embed_text for c in manifest.chunks])
    stamp = IndexStamp(
        embedder=embedder.name,
        embedding_fingerprint=embedder.fingerprint,
        chunk_manifest_sha256=sha,
        corpus_fingerprint=manifest.corpus_fingerprint,
    )
    index = ChunkIndex(config.resolve(config.index.path), config.index)
    report = index.sync(manifest.chunks, vectors, stamp)
    return IngestReport(
        chunk_count=manifest.chunk_count,
        manifest_path=path,
        manifest_sha256=sha,
        manifest_changed=previous != manifest_bytes(manifest),
        sync=report,
    )


def check_manifest(config: RagConfig, counter: TokenCounter, *, tokenizer_fingerprint: str) -> bool:
    """True if a fresh build reproduces the saved manifest byte for byte."""
    sources = load_sources(config.resolve(config.corpus.sources))
    fresh = build_chunk_manifest(
        config, sources, counter, tokenizer_fingerprint=tokenizer_fingerprint
    )
    path = config.resolve(config.chunking.manifest)
    return path.is_file() and path.read_bytes() == manifest_bytes(fresh)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="verify the saved manifest; write nothing"
    )
    args = parser.parse_args(argv)
    config = RagConfig.load()
    settings = config.embedding
    # Report every missing input at once, not just the first.
    problems: list[str] = []
    try:
        sources = load_sources(config.resolve(config.corpus.sources))
        verify_snapshot(sources, config.resolve(config.corpus.dir))
    except RagError as exc:
        problems.append(str(exc))
    counter: WordPieceCounter | OnnxEmbedder | None = None
    try:
        if args.check:
            counter = WordPieceCounter.from_settings(settings, config.root)
        else:
            counter = OnnxEmbedder(settings, config.root)
    except RagError as exc:
        problems.append(str(exc))
    if problems or counter is None:
        print("ingest cannot run:\n- " + "\n- ".join(problems), file=sys.stderr)
        return 1
    try:
        if isinstance(counter, WordPieceCounter):
            same = check_manifest(
                config, counter, tokenizer_fingerprint=settings.tokenizer_fingerprint()
            )
            print("chunk manifest reproduced byte for byte" if same else "chunk manifest DIFFERS")
            return 0 if same else 1
        report = ingest(config, counter, tokenizer_fingerprint=settings.tokenizer_fingerprint())
    except RagError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    print(f"chunks:          {report.chunk_count}")
    print(f"chunk manifest:  {report.manifest_path} (sha256 {report.manifest_sha256})")
    print(f"manifest changed since last ingest: {report.manifest_changed}")
    print(
        f"index:           {report.sync.total} vectors; upserted {report.sync.upserted}, "
        f"deleted {report.sync.deleted}, replaced collection: {report.sync.replaced_collection}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
