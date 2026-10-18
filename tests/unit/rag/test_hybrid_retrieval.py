"""Hybrid ranking: the embedding and keyword rankings fused by reciprocal rank, cited
sections first, and each retrieved chunk recording why it was returned."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest

from apps.rag.config import RagConfig, RetrievalSettings
from apps.rag.embedding import HashingEmbedder
from apps.rag.errors import RagError, StaleIndexError
from apps.rag.index import IndexStamp
from apps.rag.lexical import Analyzer
from apps.rag.rerank import RerankScore
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
        {"version": SHIPPED.version + 1},
        {"candidates": 10},
        {"rrf_k": 1},
        {"bm25": {"k1": 1.5, "b": 0.75}},
    ):
        assert _settings(**change).fingerprint() != base, change


def test_an_index_returning_unknown_chunks_is_refused_even_outside_the_top_k() -> None:
    retriever = _retriever(FixedIndex([*IDS[:3], "synthetic-part99/99.999"]))
    with pytest.raises(StaleIndexError, match="manifest lacks"):
        retriever.retrieve("loss management", top_k=1)


CHUNKS = {c.id: c for c in MANIFEST.chunks}


class ScriptedReranker:
    """Scores given per chunk (0 for the rest); records the passages it was shown."""

    name = "scripted-reranker"

    def __init__(
        self, scores: dict[str, float], *, fingerprint: str = "r" * 64, cut: int = 0
    ) -> None:
        self._by_text = {CHUNKS[cid].embed_text: s for cid, s in scores.items()}
        self.fingerprint = fingerprint
        self.cut = cut
        self.shown: list[list[str]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[RerankScore]:
        self.shown.append(list(passages))
        return [RerankScore(self._by_text.get(p, 0.0), self.cut) for p in passages]


def _reranking(index: VectorIndex, reranker: ScriptedReranker, **changes: Any) -> Retriever:
    return Retriever(
        embedder=EMBEDDER,
        index=index,
        manifest=MANIFEST,
        manifest_sha256="m" * 64,
        settings=_settings(mode="hybrid_rerank", **changes),
        reranker=reranker,
    )


def test_rerank_mode_orders_the_fused_pool_by_reranker_score() -> None:
    reranker = ScriptedReranker({_id("99.10(a)"): 3.0, _id("99.300(e)"): 2.0}, cut=7)
    record = _reranking(FixedIndex(IDS), reranker).retrieve("loss management", top_k=3)
    assert (record.mode, record.reranker, record.reranker_fingerprint) == (
        "hybrid_rerank",
        "scripted-reranker",
        "r" * 64,
    )
    first, second, third = record.results
    assert (first.chunk_id, first.score, first.rerank_score) == (_id("99.10(a)"), 3.0, 3.0)
    assert first.fused_score is not None
    assert math.isclose(first.fused_score, 1 / (60 + 6))  # 6th by embedding, no keyword
    assert second.chunk_id == _id("99.300(e)")
    # Equal reranker scores fall back to the fused order: 99.300(c) was fused first.
    assert (third.chunk_id, third.rerank_score) == (_id("99.300(c)"), 0.0)
    assert all(r.rerank_truncated_tokens == 7 for r in record.results)


def test_the_reranker_sees_the_top_fused_chunks_plus_cited_ones_and_citations_win() -> None:
    reranker = ScriptedReranker({IDS[0]: 5.0, _id("99.300(c)"): 4.0})
    retriever = _reranking(FixedIndex(IDS), reranker, candidates=2)
    record = retriever.retrieve("loss management rule under 99.30", top_k=5)
    # Fused top 2: the embedding's first chunk and the keyword ranking's first. The
    # cited 99.30 is below them, so it joins the pool on its own.
    assert reranker.shown == [
        [CHUNKS[cid].embed_text for cid in (IDS[0], _id("99.300(c)"), _id("99.30"))]
    ]
    assert [r.chunk_id for r in record.results] == [_id("99.30"), IDS[0], _id("99.300(c)")]
    assert record.results[0].cited_as == "99.30"


def test_rerank_mode_needs_a_reranker_and_its_fingerprint_counts() -> None:
    with pytest.raises(RagError, match="needs a reranker"):
        _retriever(FixedIndex(IDS), mode="hybrid_rerank")
    one = _reranking(FixedIndex(IDS), ScriptedReranker({}, fingerprint="1" * 64))
    other = _reranking(FixedIndex(IDS), ScriptedReranker({}, fingerprint="2" * 64))
    assert len({one.fingerprint, other.fingerprint, SHIPPED.fingerprint()}) == 3
    # A reranker passed to a mode that does not use it is ignored, not recorded.
    hybrid = Retriever(
        embedder=EMBEDDER,
        index=FixedIndex(IDS),
        manifest=MANIFEST,
        manifest_sha256="m" * 64,
        settings=SHIPPED,
        reranker=ScriptedReranker({}),
    )
    record = hybrid.retrieve("loss management")
    assert (hybrid.fingerprint, record.reranker) == (SHIPPED.fingerprint(), None)
    assert all(r.rerank_score is None and r.fused_score == r.score for r in record.results)


def test_feedback_lets_the_keyword_ranking_match_words_the_question_lacks() -> None:
    query = "zzzq qqqz"  # no term in the corpus
    plain = _retriever(FixedIndex(IDS)).retrieve(query, top_k=5)
    assert all(r.lexical_rank is None for r in plain.results)
    assert plain.query_expansion is None
    expanded = _retriever(FixedIndex(IDS), mode="hybrid_prf").retrieve(query, top_k=5)
    assert expanded.mode == "hybrid_prf"
    assert expanded.query_expansion is not None
    assert {"zzzq", "qqqz"} <= expanded.query_expansion.keys()
    assert len(expanded.query_expansion) <= 2 + SHIPPED.feedback.terms
    assert any(r.lexical_rank is not None for r in expanded.results)


def test_feedback_comes_from_the_embedding_ranking_top_chunks() -> None:
    index = FixedIndex(IDS)
    one = _retriever(
        index, mode="hybrid_prf", feedback={**SHIPPED.feedback.model_dump(), "docs": 1}
    )
    record = one.retrieve("zzzq", top_k=3)
    assert record.query_expansion is not None
    feedback_terms = set(record.query_expansion) - {"zzzq"}
    first = Analyzer().terms(CHUNKS[IDS[0]].embed_text)
    assert feedback_terms <= set(first)  # all from the embedding's first chunk


def test_feedback_settings_count_only_in_the_feedback_mode() -> None:
    changed = {**SHIPPED.feedback.model_dump(), "terms": 3}
    assert _settings(feedback=changed).fingerprint() == SHIPPED.fingerprint()
    prf = _settings(mode="hybrid_prf")
    assert prf.fingerprint() != SHIPPED.fingerprint()
    assert _settings(mode="hybrid_prf", feedback=changed).fingerprint() != prf.fingerprint()
