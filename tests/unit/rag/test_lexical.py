"""BM25 keyword scoring: the analyzer, the formula and the ordering."""

from __future__ import annotations

import math

import pytest

from apps.rag.lexical import Analyzer, Bm25Index


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
