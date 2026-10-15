"""Retrieve the top-k chunks for a query and describe them as saved evidence.

Two modes (config/rag.yaml, retrieval.mode):

- dense: the chunks whose embeddings are closest to the query's (cosine).
- hybrid: the embedding ranking and a BM25 keyword ranking (apps.rag.lexical), each
  cut to its top `candidates`, fused by reciprocal rank: a chunk scores
  1 / (rrf_k + rank) in each ranking it appears in. Then the chunks the question
  cites by section number (apps.rag.citations) move to the front: for each citation
  in order, its best-scoring chunk, then the rest of the cited chunks by score.
  Ties are broken by embedding similarity, then by chunk ID.

Each retrieved chunk records its score, where each ranking placed it and the
citation that points at it, so a record shows why a chunk was returned.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from apps.core.evidence import RetrievalRecord, RetrievedChunkRecord
from apps.core.tracing.base import Tracer
from apps.core.tracing.local import NoopTracer
from apps.rag.chunking import Chunk, ChunkManifest, read_manifest
from apps.rag.citations import Citation, parse_citations, points_at
from apps.rag.config import RagConfig, RetrievalSettings
from apps.rag.corpus import load_sources, verify_snapshot
from apps.rag.embedding import Embedder
from apps.rag.errors import StaleIndexError
from apps.rag.index import ChunkIndex, IndexStamp
from apps.rag.lexical import Bm25Index


class VectorIndex(Protocol):
    """What the retriever needs from an index (ChunkIndex; tests use an in-memory one)."""

    def query(
        self, vector: Sequence[float], k: int, expected: IndexStamp
    ) -> list[tuple[str, float]]: ...


@dataclass
class _Candidate:
    chunk_id: str
    score: float = 0.0
    dense_similarity: float | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    cited_as: str | None = None

    def order(self) -> tuple[float, float, str]:
        similarity = self.dense_similarity if self.dense_similarity is not None else -math.inf
        return (-self.score, -similarity, self.chunk_id)


class Retriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        index: VectorIndex,
        manifest: ChunkManifest,
        manifest_sha256: str,
        settings: RetrievalSettings,
        tracer: Tracer | None = None,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.settings = settings
        self.top_k = settings.top_k
        self.fingerprint = settings.fingerprint()
        self.tracer = tracer or NoopTracer()
        self._chunks = {c.id: c for c in manifest.chunks}
        self._lexical = (
            Bm25Index(
                [(c.id, c.embed_text) for c in manifest.chunks],
                k1=settings.bm25.k1,
                b=settings.bm25.b,
            )
            if settings.mode == "hybrid"
            else None
        )
        self.stamp = IndexStamp(
            embedder=embedder.name,
            embedding_fingerprint=embedder.fingerprint,
            chunk_manifest_sha256=manifest_sha256,
            corpus_fingerprint=manifest.corpus_fingerprint,
        )

    def chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def _dense(self, vector: Sequence[float], k: int) -> list[_Candidate]:
        return [
            _Candidate(chunk_id, score=similarity, dense_similarity=similarity, dense_rank=rank)
            for rank, (chunk_id, similarity) in enumerate(
                self.index.query(vector, k, self.stamp), start=1
            )
        ]

    def _hybrid(
        self,
        lexical: Bm25Index,
        query: str,
        vector: Sequence[float],
        k: int,
        citations: list[Citation],
    ) -> list[_Candidate]:
        depth = min(self.settings.candidates, len(self._chunks))
        rrf_k = self.settings.rrf_k
        candidates: dict[str, _Candidate] = {}
        for rank, (chunk_id, similarity) in enumerate(
            self.index.query(vector, depth, self.stamp), start=1
        ):
            if chunk_id not in self._chunks:
                raise StaleIndexError(f"index returned {chunk_id}, which the manifest lacks")
            c = candidates.setdefault(chunk_id, _Candidate(chunk_id))
            c.dense_similarity, c.dense_rank = similarity, rank
            c.score += 1.0 / (rrf_k + rank)
        for rank, (chunk_id, score) in enumerate(lexical.search(query, depth), start=1):
            c = candidates.setdefault(chunk_id, _Candidate(chunk_id))
            c.lexical_score, c.lexical_rank = score, rank
            c.score += 1.0 / (rrf_k + rank)
        cited: dict[Citation, list[str]] = {}
        for citation in citations:
            cited[citation] = []
            for chunk in self.manifest.chunks:
                if points_at(citation, chunk.doc_kind, chunk.locator):
                    cited[citation].append(chunk.id)
                    c = candidates.setdefault(chunk.id, _Candidate(chunk.id))
                    c.cited_as = c.cited_as or citation.locator
        ranked = sorted(candidates.values(), key=_Candidate.order)
        first: list[_Candidate] = []
        for citation in citations:
            best = next(
                (c for c in ranked if c.chunk_id in cited[citation] and c not in first), None
            )
            if best is not None:
                first.append(best)
        first += [c for c in ranked if c.cited_as is not None and c not in first]
        return (first + [c for c in ranked if c.cited_as is None])[:k]

    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalRecord:
        k = top_k or self.top_k
        mode = self.settings.mode
        lexical = self._lexical  # built only in hybrid mode
        with self.tracer.observe(
            "rag.retrieve",
            kind="retriever",
            input={"query": query, "top_k": k},
            metadata={
                "mode": mode,
                "retrieval_fingerprint": self.fingerprint,
                "embedder": self.embedder.name,
                "embedding_fingerprint": self.embedder.fingerprint,
                "chunk_manifest_sha256": self.manifest_sha256,
            },
        ) as observation:
            vector = self.embedder.embed_query(query)
            citations = parse_citations(query) if lexical is not None else []
            ranked = (
                self._hybrid(lexical, query, vector, k, citations)
                if lexical is not None
                else self._dense(vector, k)
            )
            results: list[RetrievedChunkRecord] = []
            for rank, candidate in enumerate(ranked, start=1):
                chunk = self._chunks.get(candidate.chunk_id)
                if chunk is None:
                    raise StaleIndexError(
                        f"index returned {candidate.chunk_id}, which the manifest lacks"
                    )
                results.append(
                    RetrievedChunkRecord(
                        rank=rank,
                        chunk_id=chunk.id,
                        doc_id=chunk.doc_id,
                        doc_kind=chunk.doc_kind,
                        label=chunk.label,
                        heading_path=list(chunk.heading_path),
                        score=candidate.score,
                        dense_similarity=candidate.dense_similarity,
                        dense_rank=candidate.dense_rank,
                        lexical_score=candidate.lexical_score,
                        lexical_rank=candidate.lexical_rank,
                        cited_as=candidate.cited_as,
                        text=chunk.text,
                        text_sha256=chunk.text_sha256,
                    )
                )
            observation.update(
                output=[
                    {"chunk_id": r.chunk_id, "score": r.score, "rank": r.rank, "cited": r.cited_as}
                    for r in results
                ]
            )
            trace_id = observation.trace_id
        return RetrievalRecord(
            query=query,
            top_k=k,
            mode=mode,
            retrieval_fingerprint=self.fingerprint,
            citations=[c.locator for c in citations],
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
        settings=config.retrieval,
        tracer=tracer,
    )
