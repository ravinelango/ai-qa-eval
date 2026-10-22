"""Deterministic scoring of one answer against one golden case.

Pure functions, no I/O: the same answer always gets the same score. A case
passes only if every check passes. Phrase checks compare case-insensitively
after collapsing whitespace and straightening quotes; `must_match` regexes run
on the raw answer with re.IGNORECASE.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from apps.core.types import StopReason
from evals.golden.dataset import GoldenCase

CheckKind = Literal["must_include", "must_not_include", "must_match", "decline", "not_refused"]

_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: CheckKind
    expected: str
    passed: bool


def normalize(text: str) -> str:
    return " ".join(text.translate(_QUOTES).casefold().split())


def _contains(haystack: str, phrase: str) -> bool:
    return normalize(phrase) in haystack


def score_answer(
    case: GoldenCase,
    text: str,
    stop_reason: StopReason,
    decline_markers: tuple[str, ...],
) -> list[CheckResult]:
    answer = normalize(text)
    checks: list[CheckResult] = []
    expect = case.expect

    for phrase in expect.must_include:
        options = (phrase,) if isinstance(phrase, str) else phrase
        checks.append(
            CheckResult(
                kind="must_include",
                expected=" | ".join(options),
                passed=any(_contains(answer, option) for option in options),
            )
        )
    for phrase in expect.must_not_include:
        checks.append(
            CheckResult(
                kind="must_not_include", expected=phrase, passed=not _contains(answer, phrase)
            )
        )
    for pattern in expect.must_match:
        checks.append(
            CheckResult(
                kind="must_match",
                expected=pattern,
                passed=re.search(pattern, text, re.IGNORECASE) is not None,
            )
        )

    refused = stop_reason == "refusal"
    if expect.expect_decline:
        declined = refused or any(_contains(answer, marker) for marker in decline_markers)
        checks.append(CheckResult(kind="decline", expected="answer declines", passed=declined))
    else:
        # Without this, a refusal would pass a case whose only checks are must_not_include.
        checks.append(
            CheckResult(kind="not_refused", expected="no provider refusal", passed=not refused)
        )
    return checks
