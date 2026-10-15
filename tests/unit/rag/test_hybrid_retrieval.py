"""Hybrid ranking: the embedding and keyword rankings fused by reciprocal rank, cited
sections first, and each retrieved chunk recording why it was returned."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest

from apps.rag.config import RagConfig, RetrievalSettings
from apps.rag.embedding import HashingEmbedder
from apps.rag.errors import StaleIndexError
from apps.rag.index import IndexStamp
from apps.rag.retrieve import Retriever, VectorIndex
from tests.rag_helpers import InMemoryIndex, part99_manifest

EMBEDDER = HashingEmbedder(max_tokens=128)
MANIFEST = part99_manifest(EMBEDDER)
IDS = [c.id for c in MANIFEST.chunks]
STAMP = IndexStamp(EMBEDDER.name, EMBEDDER.fingerprint, "m" * 64, "c" * 64)
SHIPPED = RagConfig.load().retrieval


def _id(locator: str) -> str:
    return f"synthetic-part99/{locator}"


class FixedIndex:
    """A fixed embedding ranking (similarity 0.9, 0.89, ...); records the depth asked."""

    def __init__(self, ranking: Sequence[str]) -> None:
        self.ranking = list(ranking)
        self.depths: list[int] = []

    def query(
        self, vector: Sequence[float], k: int, expected: IndexStamp
    ) -> list[tuple[str, float]]:
        assert expected == STAMP
        self.depths.append(k)
        return [(cid, 0.9 - i / 100) for i, cid in enumerate(self.ranking[:k])]


def _settings(**changes: Any) -> RetrievalSettings:
    return RetrievalSettings.model_validate({**SHIPPED.model_dump(), **changes})


def _retriever(index: VectorIndex, **changes: Any) -> Retriever:
    return Retriever(
        embedder=EMBEDDER,
        index=index,
        manifest=MANIFEST,
        manifest_sha256="m" * 64,
        settings=_settings(**changes),
    )


def test_dense_mode_is_the_embedding_ranking_alone() -> None:
    index = InMemoryIndex(STAMP)
    index.add(MANIFEST.chunks, EMBEDDER.embed_documents([c.embed_text for c in MANIFEST.chunks]))
    query = "What does 99.30 say about placeholder change logs?"
    record = _retriever(index, mode="dense").retrieve(query, top_k=3)
    expected = index.query(EMBEDDER.embed_query(query), 3, STAMP)
    assert [(r.chunk_id, r.score) for r in record.results] == expected
    assert (record.mode, record.citations) == ("dense", [])
    for rank, r in enumerate(record.results, start=1):
        assert (r.dense_similarity, r.dense_rank) == (r.score, rank)
        assert (r.lexical_score, r.lexical_rank, r.cited_as) == (None, None, None)


def test_a_keyword_match_lifts_a_chunk_the_embedding_ranks_low() -> None:
    index = FixedIndex(IDS)  # manifest order: 99.300(c) is 25th
    record = _retriever(index).retrieve("loss management", top_k=3)
    top, second = record.results[0], record.results[1]
    assert top.chunk_id == _id("99.300(c)")
    assert (top.dense_rank, top.lexical_rank) == (25, 1)
    assert math.isclose(top.score, 1 / (60 + 25) + 1 / (60 + 1))
    assert (second.chunk_id, second.dense_rank, second.lexical_rank) == (IDS[0], 1, None)
    assert math.isclose(second.score, 1 / (60 + 1))
    # Each ranking contributes at most `candidates` chunks, never more than exist.
    assert index.depths == [len(IDS)]
    _retriever(index, candidates=5).retrieve("loss management")
    assert index.depths[-1] == 5


def test_ties_break_by_embedding_similarity_then_id() -> None:
    # Only the embedding's top chunk and the keyword's top chunk are candidates, and
    # each is first in one ranking: equal fused scores.
    record = _retriever(FixedIndex(IDS), candidates=1).retrieve("loss management")
    assert [r.chunk_id for r in record.results] == [IDS[0], _id("99.300(c)")]
    assert record.results[0].score == record.results[1].score


def test_cited_sections_come_first_best_match_per_citation() -> None:
    index = FixedIndex(list(reversed(IDS)))
    record = _retriever(index).retrieve("What do 99.50 and 99.10(c) require?", top_k=5)
    assert record.citations == ["99.50", "99.10(c)"]
    first, second, third = record.results[:3]
    assert {first.chunk_id, third.chunk_id} == {_id("99.50(a)"), _id("99.50(b)")}
    assert first.score >= third.score
    assert (first.cited_as, third.cited_as) == ("99.50", "99.50")
    assert (second.chunk_id, second.cited_as) == (_id("99.10(c)"), "99.10(c)")
    assert all(r.cited_as is None for r in record.results[3:])


def test_a_cited_chunk_outside_both_rankings_still_comes_first() -> None:
    record = _retriever(FixedIndex(IDS), candidates=1).retrieve("loss management rule under 99.30")
    top = record.results[0]
    assert (top.chunk_id, top.cited_as) == (_id("99.30"), "99.30")
    assert (top.dense_rank, top.lexical_rank, top.score) == (None, None, 0.0)


def test_the_record_says_how_chunks_were_ranked() -> None:
    record = _retriever(FixedIndex(IDS)).retrieve("Summarize 99.300(e).", top_k=2)
    assert record.mode == "hybrid"
    assert record.retrieval_fingerprint == SHIPPED.fingerprint()
    assert record.citations == ["99.300(e)"]
    assert record.results[0].chunk_id == _id("99.300(e)")
    assert record.results[0].lexical_score is not None


def test_the_fingerprint_tracks_everything_but_top_k() -> None:
    base = SHIPPED.fingerprint()
    assert _settings(top_k=9).fingerprint() == base
    for change in (
        {"mode": "dense"},
        {"version": 2},
        {"candidates": 10},
        {"rrf_k": 1},
        {"bm25": {"k1": 1.5, "b": 0.75}},
    ):
        assert _settings(**change).fingerprint() != base, change


def test_an_index_returning_unknown_chunks_is_refused_even_outside_the_top_k() -> None:
    retriever = _retriever(FixedIndex([*IDS[:3], "synthetic-part99/99.999"]))
    with pytest.raises(StaleIndexError, match="manifest lacks"):
        retriever.retrieve("loss management", top_k=1)
