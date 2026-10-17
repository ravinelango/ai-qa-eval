"""Keyword scoring (Okapi BM25) over the chunk manifest.

Each chunk is scored on the text that is embedded for it (heading path, label and
text). Text is lowercased, a possessive "'s" is dropped, the rest is split into runs
of letters and digits, Lucene's default English stop words are removed and each
remaining word is reduced by the Snowball English (Porter2) stemmer, so "signs",
"signing" and "signings" match each other. Scores follow the classic BM25 formula
with Lucene's always-positive IDF; k1 and b come from config/rag.yaml (Lucene's
defaults). Each distinct query term counts once.

The index is built in memory from the chunk manifest when the retriever starts. It
holds nothing that is not derived from the manifest, so it cannot drift from it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import snowballstemmer

# Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET.
STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
        "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
        "there", "these", "they", "this", "to", "was", "will", "with",
    }
)  # fmt: skip
_POSSESSIVE = re.compile(r"['\u2019]s\b")
_WORD = re.compile(r"[^\W_]+")


class Analyzer:
    """Turns text into the terms BM25 counts."""

    def __init__(self) -> None:
        self._stemmer = snowballstemmer.stemmer("english")

    def terms(self, text: str) -> list[str]:
        words = _WORD.findall(_POSSESSIVE.sub("", text.lower()))
        stems: list[str] = self._stemmer.stemWords([w for w in words if w not in STOP_WORDS])
        return stems


class Bm25Index:
    def __init__(
        self,
        documents: Sequence[tuple[str, str]],
        *,
        k1: float,
        b: float,
        analyzer: Analyzer | None = None,
    ) -> None:
        """`documents` are (ID, text) pairs; IDs must be unique."""
        self.analyzer = analyzer or Analyzer()
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._counts: list[Counter[str]] = []
        self._lengths: list[int] = []
        frequency: Counter[str] = Counter()
        for doc_id, text in documents:
            counts = Counter(self.analyzer.terms(text))
            self._ids.append(doc_id)
            self._counts.append(counts)
            self._lengths.append(sum(counts.values()))
            frequency.update(counts.keys())
        if len(set(self._ids)) != len(self._ids):
            raise ValueError("document IDs must be unique")
        n = len(self._ids)
        self._average_length = sum(self._lengths) / n if n else 0.0
        self._idf = {
            term: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for term, df in frequency.items()
        }

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """The top-k (ID, score), best first, ties broken by ID. Documents that share
        no term with the query are left out."""
        terms = sorted(set(self.analyzer.terms(query)) & self._idf.keys())
        scored: list[tuple[str, float]] = []
        for doc_id, counts, length in zip(self._ids, self._counts, self._lengths, strict=True):
            norm = self.k1 * (1.0 - self.b + self.b * length / self._average_length)
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    score += self._idf[term] * tf * (self.k1 + 1.0) / (tf + norm)
            if score > 0.0:
                scored.append((doc_id, score))
        scored.sort(key=lambda p: (-p[1], p[0]))
        return scored[:k]
