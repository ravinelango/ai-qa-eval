"""BM25 keyword scoring: the analyzer, the formula and the ordering."""

from __future__ import annotations

import math

import pytest

from apps.rag.lexical import Analyzer, Bm25Index, rocchio_expansion


def test_analyzer_lowercases_drops_stop_words_and_possessives_and_stems() -> None:
    text = "FDA\u2019s rules: the Signs, signing and SIGNINGS of records — résumé"
    terms = Analyzer().terms(text)
    assert terms == ["fda", "rule", "sign", "sign", "sign", "record", "résumé"]


def test_scores_follow_the_bm25_formula() -> None:
    index = Bm25Index(
        [("a", "audit trail audit"), ("b", "password review"), ("c", "audit password")],
        k1=1.2,
        b=0.75,
    )
    # "audit" is in 2 of the 3 documents; document a has it twice in 3 terms.
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    norm = 1.2 * (1 - 0.75 + 0.75 * 3 / ((3 + 2 + 2) / 3))
    hits = dict(index.search("audit", 10))
    assert hits.keys() == {"a", "c"}  # b shares no term with the query
    assert math.isclose(hits["a"], idf * 2 * 2.2 / (2 + norm))


def test_rare_terms_outweigh_common_ones_and_ties_break_by_id() -> None:
    index = Bm25Index(
        [("z", "record change"), ("y", "record change"), ("x", "record continuous")],
        k1=1.2,
        b=0.75,
    )
    assert index.search("continuous record", 10)[0][0] == "x"
    assert [doc for doc, _ in index.search("change", 10)] == ["y", "z"]
    assert index.search("change", 1) == index.search("change", 10)[:1]
    assert index.search("the of and", 10) == []  # stop words only
    assert index.search("unrelated", 10) == []


def test_document_ids_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        Bm25Index([("a", "x"), ("a", "y")], k1=1.2, b=0.75)


def test_weighted_search_with_unit_weights_is_plain_search() -> None:
    index = Bm25Index(
        [("a", "audit trail audit"), ("b", "password review"), ("c", "audit password")],
        k1=1.2,
        b=0.75,
    )
    assert index.search_weighted({"audit": 1.0, "password": 1.0}, 10) == index.search(
        "audit password", 10
    )
    doubled = dict(index.search_weighted({"audit": 2.0}, 10))
    assert math.isclose(doubled["a"], 2 * dict(index.search("audit", 10))["a"])


def test_rocchio_expansion_adds_the_feedback_documents_strongest_terms() -> None:
    index = Bm25Index(
        [("a", "audit trail audit trail log"), ("b", "password review"), ("c", "shared login")],
        k1=1.2,
        b=0.75,
    )
    weights = rocchio_expansion(index, "change history", ["a"], terms=2, alpha=1.0, beta=0.75)
    # Query terms share alpha; "audit" and "trail" are each 2/5 of document a; "log"
    # (1/5) is cut by terms=2.
    assert weights == {"chang": 0.5, "histori": 0.5, "audit": 0.75 * 0.4, "trail": 0.75 * 0.4}
    # No query terms: only feedback. Averaged over a and b, "password" and "review"
    # (1/2 of b, so 1/4) beat "audit" (2/5 of a, so 1/5); the tie breaks by term.
    assert rocchio_expansion(index, "the of", ["a", "b"], terms=1, alpha=1.0, beta=1.0) == {
        "password": 0.25
    }
