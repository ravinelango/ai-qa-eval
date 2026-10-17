"""The real Chroma client, offline: idempotent sync, stale-index refusal, scores."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.rag.chunking import Chunk
from apps.rag.config import RagConfig
from apps.rag.errors import StaleIndexError
from apps.rag.index import ChunkIndex, IndexStamp

STAMP = IndexStamp(
    embedder="test",
    embedding_fingerprint="e" * 64,
    chunk_manifest_sha256="m" * 64,
    corpus_fingerprint="c" * 64,
)


def _chunk(cid: str) -> Chunk:
    return Chunk(
        id=f"doc/{cid}",
        doc_id="doc",
        doc_title="Doc",
        doc_kind="regulation",
        domain="test",
        locator=cid,
        label=cid,
        heading_path=("H",),
        paragraphs=(),
        pages=(),
        elements=(),
        part=1,
        parts=1,
        text=f"text {cid}",
        token_count=4,
        text_sha256="0" * 64,
    )


CHUNKS = [_chunk("a"), _chunk("b"), _chunk("c")]
VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.8, 0.0]]


def _index(tmp_path: Path) -> ChunkIndex:
    return ChunkIndex(tmp_path / "chroma", RagConfig.load().index)


def test_sync_is_idempotent_and_never_duplicates(tmp_path: Path) -> None:
    index = _index(tmp_path)
    first = index.sync(CHUNKS, VECTORS, STAMP)
    second = index.sync(CHUNKS, VECTORS, STAMP)
    assert (first.total, second.total) == (3, 3)
    assert (second.deleted, second.replaced_collection) == (0, False)
    # A fresh client on the same directory sees the same index.
    assert _index(tmp_path).stamp() == STAMP


def test_chunks_dropped_from_the_manifest_are_deleted(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.sync(CHUNKS, VECTORS, STAMP)
    report = index.sync(CHUNKS[:2], VECTORS[:2], STAMP)
    assert (report.deleted, report.total) == (1, 2)


def test_a_new_embedding_model_replaces_the_collection(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.sync(CHUNKS, VECTORS, STAMP)
    other = IndexStamp("other", "f" * 64, STAMP.chunk_manifest_sha256, STAMP.corpus_fingerprint)
    report = index.sync(CHUNKS[:1], VECTORS[:1], other)
    assert (report.replaced_collection, report.total) == (True, 1)


def test_query_returns_cosine_similarity_best_first(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.sync(CHUNKS, VECTORS, STAMP)
    hits = index.query([1.0, 0.0, 0.0], 3, STAMP)
    assert [cid for cid, _ in hits] == ["doc/a", "doc/c", "doc/b"]
    scores = [score for _, score in hits]
    assert scores == pytest.approx([1.0, 0.6, 0.0], abs=1e-5)


def test_an_interrupted_sync_leaves_an_index_that_is_refused_then_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chromadb.api.models.Collection import Collection

    index = _index(tmp_path)

    def broken_upsert(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(Collection, "upsert", broken_upsert)
        with pytest.raises(RuntimeError, match="interrupted"):
            index.sync(CHUNKS, VECTORS, STAMP)
    assert index.stamp() is None
    with pytest.raises(StaleIndexError, match="different inputs"):
        index.query([1.0, 0.0, 0.0], 1, STAMP)
    report = index.sync(CHUNKS, VECTORS, STAMP)
    assert (report.replaced_collection, report.total) == (True, 3)
    assert index.query([1.0, 0.0, 0.0], 1, STAMP)[0][0] == "doc/a"


def test_query_refuses_a_missing_or_stale_index(tmp_path: Path) -> None:
    index = _index(tmp_path)
    with pytest.raises(StaleIndexError, match="no index"):
        index.query([1.0, 0.0, 0.0], 1, STAMP)
    index.sync(CHUNKS, VECTORS, STAMP)
    edited = IndexStamp(STAMP.embedder, STAMP.embedding_fingerprint, "n" * 64, "c" * 64)
    with pytest.raises(StaleIndexError, match="different inputs"):
        index.query([1.0, 0.0, 0.0], 1, edited)
