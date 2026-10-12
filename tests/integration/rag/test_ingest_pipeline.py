"""End to end on a synthetic frozen corpus: verify, parse, chunk, manifest, embed
(hashing test embedder), Chroma, retrieve. No network, no real model."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.rag.config import RagConfig
from apps.rag.corpus import load_sources
from apps.rag.embedding import HashingEmbedder
from apps.rag.errors import CorpusError, StaleIndexError
from apps.rag.ingest import build_chunk_manifest, check_manifest, ingest
from apps.rag.retrieve import build_retriever
from tests.rag_helpers import synthetic_corpus

EMBEDDER = HashingEmbedder(max_tokens=64)


@pytest.fixture
def config(tmp_path: Path) -> RagConfig:
    return synthetic_corpus(tmp_path)


def test_ingest_twice_gives_identical_manifest_and_no_duplicates(config: RagConfig) -> None:
    first = ingest(config, EMBEDDER, tokenizer_fingerprint=EMBEDDER.fingerprint)
    manifest_bytes = first.manifest_path.read_bytes()
    second = ingest(config, EMBEDDER, tokenizer_fingerprint=EMBEDDER.fingerprint)

    assert second.manifest_path.read_bytes() == manifest_bytes
    assert second.manifest_sha256 == first.manifest_sha256
    assert second.manifest_changed is False
    assert second.sync.total == first.sync.total == first.chunk_count
    assert second.sync.deleted == 0
    assert check_manifest(config, EMBEDDER, tokenizer_fingerprint=EMBEDDER.fingerprint)


def test_manifest_records_kind_domain_and_locations(config: RagConfig) -> None:
    sources = load_sources(config.resolve(config.corpus.sources))
    manifest = build_chunk_manifest(
        config, sources, EMBEDDER, tokenizer_fingerprint=EMBEDDER.fingerprint
    )
    by_id = {c.id: c for c in manifest.chunks}
    reg = by_id["synthetic-part99/99.10(e)"]
    assert (reg.doc_kind, reg.label, reg.elements) == (
        "regulation",
        "§99.10(e)",
        ("DIV8[§ 99.10]/P[6]",),
    )
    guidance = by_id["synthetic-guidance/II.A"]
    assert (guidance.doc_kind, guidance.pages) == ("guidance", (3,))
    assert guidance.domain == "Synthetic ledgers (test only)"
    assert manifest.corpus_fingerprint == sources.fingerprint()
    assert all(c.token_count <= EMBEDDER.max_tokens for c in manifest.chunks)


def test_ingest_refuses_a_corpus_file_that_changed(config: RagConfig) -> None:
    path = config.resolve(config.corpus.dir) / "part99.xml"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(CorpusError, match="has SHA-256"):
        ingest(config, EMBEDDER, tokenizer_fingerprint=EMBEDDER.fingerprint)


def test_retrieval_over_the_ingested_index(config: RagConfig) -> None:
    ingest(config, EMBEDDER, tokenizer_fingerprint=EMBEDDER.fingerprint)
    retriever = build_retriever(config, EMBEDDER, tokenizer_fingerprint=EMBEDDER.fingerprint)
    record = retriever.retrieve("placeholder change logs that record the earlier value", top_k=3)
    assert record.results[0].chunk_id == "synthetic-part99/99.10(e)"
    assert [r.rank for r in record.results] == [1, 2, 3]
    assert record.results[0].score >= record.results[1].score
    assert record.results[0].text.startswith("(e) Use of placeholder change logs")
    assert record.chunk_manifest_sha256 == retriever.manifest_sha256


def test_retriever_refuses_an_index_built_with_another_embedder(config: RagConfig) -> None:
    ingest(config, EMBEDDER, tokenizer_fingerprint=EMBEDDER.fingerprint)
    other = HashingEmbedder(dimensions=64, max_tokens=64)
    retriever = build_retriever(config, other, tokenizer_fingerprint=EMBEDDER.fingerprint)
    with pytest.raises(StaleIndexError, match="different inputs"):
        retriever.retrieve("anything")


def test_retriever_refuses_a_manifest_older_than_the_corpus_or_chunker(config: RagConfig) -> None:
    ingest(config, EMBEDDER, tokenizer_fingerprint=EMBEDDER.fingerprint)
    sources = config.resolve(config.corpus.sources)
    original = sources.read_text(encoding="utf-8")
    # e.g. `make corpus-refresh` changed the corpus, but nobody re-ran ingest.
    sources.write_text(original.replace("test only", "test only, revised"), encoding="utf-8")
    with pytest.raises(StaleIndexError, match=r"changed: the corpus \(sources\.yaml\)"):
        build_retriever(config, EMBEDDER, tokenizer_fingerprint=EMBEDDER.fingerprint)
    sources.write_text(original, encoding="utf-8")

    bumped = config.model_copy(
        update={"chunking": config.chunking.model_copy(update={"version": 2})}
    )
    with pytest.raises(StaleIndexError, match=r"chunking\.version"):
        build_retriever(bumped, EMBEDDER, tokenizer_fingerprint=EMBEDDER.fingerprint)
    with pytest.raises(StaleIndexError, match="the tokenizer"):
        build_retriever(config, EMBEDDER, tokenizer_fingerprint="other")
