"""Retrieve the top-k chunks for a query and describe them as saved evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from apps.core.evidence import RetrievalRecord, RetrievedChunkRecord
from apps.core.tracing.base import Tracer
from apps.core.tracing.local import NoopTracer
from apps.rag.chunking import Chunk, ChunkManifest, read_manifest
from apps.rag.config import RagConfig
from apps.rag.corpus import load_sources, verify_snapshot
from apps.rag.embedding import Embedder
from apps.rag.errors import StaleIndexError
from apps.rag.index import ChunkIndex, IndexStamp


class VectorIndex(Protocol):
    """What the retriever needs from an index (ChunkIndex; tests use an in-memory one)."""

    def query(
        self, vector: Sequence[float], k: int, expected: IndexStamp
    ) -> list[tuple[str, float]]: ...


class Retriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        index: VectorIndex,
        manifest: ChunkManifest,
        manifest_sha256: str,
        top_k: int,
        tracer: Tracer | None = None,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.top_k = top_k
        self.tracer = tracer or NoopTracer()
        self._chunks = {c.id: c for c in manifest.chunks}
        self.stamp = IndexStamp(
            embedder=embedder.name,
            embedding_fingerprint=embedder.fingerprint,
            chunk_manifest_sha256=manifest_sha256,
            corpus_fingerprint=manifest.corpus_fingerprint,
        )

    def chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalRecord:
        k = top_k or self.top_k
        with self.tracer.observe(
            "rag.retrieve",
            kind="retriever",
            input={"query": query, "top_k": k},
            metadata={
                "embedder": self.embedder.name,
                "embedding_fingerprint": self.embedder.fingerprint,
                "chunk_manifest_sha256": self.manifest_sha256,
            },
        ) as observation:
            hits = self.index.query(self.embedder.embed_query(query), k, self.stamp)
            results: list[RetrievedChunkRecord] = []
            for rank, (chunk_id, score) in enumerate(hits, start=1):
                chunk = self._chunks.get(chunk_id)
                if chunk is None:
                    raise StaleIndexError(f"index returned {chunk_id}, which the manifest lacks")
                results.append(
                    RetrievedChunkRecord(
                        rank=rank,
                        chunk_id=chunk.id,
                        doc_id=chunk.doc_id,
                        doc_kind=chunk.doc_kind,
                        label=chunk.label,
                        heading_path=list(chunk.heading_path),
                        score=score,
                        text=chunk.text,
                        text_sha256=chunk.text_sha256,
                    )
                )
            observation.update(
                output=[{"chunk_id": r.chunk_id, "score": r.score, "rank": r.rank} for r in results]
            )
            trace_id = observation.trace_id
        return RetrievalRecord(
            query=query,
            top_k=k,
            embedder=self.embedder.name,
            embedding_fingerprint=self.embedder.fingerprint,
            chunk_manifest_sha256=self.manifest_sha256,
            results=results,
            trace_id=trace_id,
        )


def check_manifest_is_current(
    manifest: ChunkManifest, config: RagConfig, embedder: Embedder, tokenizer_fingerprint: str
) -> None:
    """Refuse a chunk manifest built from other inputs than the current configuration.

    The corpus files themselves are checked against their SHA-256 first, so an edited
    file is caught even when sources.yaml still describes the original.
    """
    sources = load_sources(config.resolve(config.corpus.sources))
    verify_snapshot(sources, config.resolve(config.corpus.dir))
    problems = [
        what
        for what, same in (
            ("the corpus (sources.yaml)", manifest.corpus_fingerprint == sources.fingerprint()),
            ("chunking.version", manifest.chunker_version == config.chunking.version),
            ("the tokenizer", manifest.tokenizer_fingerprint == tokenizer_fingerprint),
            ("the token limit", manifest.max_tokens == embedder.max_tokens),
        )
        if not same
    ]
    if problems:
        raise StaleIndexError(
            "the chunk manifest does not match the current configuration (changed: "
            f"{'; '.join(problems)}); run make ingest"
        )


def build_retriever(
    config: RagConfig,
    embedder: Embedder,
    *,
    tokenizer_fingerprint: str,
    tracer: Tracer | None = None,
) -> Retriever:
    manifest, sha = read_manifest(config.resolve(config.chunking.manifest))
    check_manifest_is_current(manifest, config, embedder, tokenizer_fingerprint)
    index = ChunkIndex(config.resolve(config.index.path), config.index)
    return Retriever(
        embedder=embedder,
        index=index,
        manifest=manifest,
        manifest_sha256=sha,
        top_k=config.retrieval.top_k,
        tracer=tracer,
    )
