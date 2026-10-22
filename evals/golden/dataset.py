"""Golden-set file format and loader.

A golden set is a YAML file: a header (name, version, synthetic flag, decline
markers) and a list of cases. Each case is a question plus deterministic
expectations about the answer. Loading validates everything up front (unique
IDs, compilable regexes, at least one expectation per case), so a bad file fails
before any model call is made.

The file's SHA-256 is recorded in every run, so a result always names the exact
dataset bytes it was scored against.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from apps.core.errors import ConfigurationError

# A required phrase is either one string or a group of accepted alternatives
# (e.g. ["time-stamped", "timestamped"]); a group passes if any member appears.
Phrase = str | tuple[str, ...]

DEFAULT_DECLINE_MARKERS = (
    "i don't know",
    "i do not know",
    "cannot answer",
    "can't answer",
    "not covered",
    "outside the scope",
    "not able to answer",
)


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Expectations(_Strict):
    """Deterministic checks. Matching is case-insensitive and whitespace-normalized."""

    must_include: tuple[Phrase, ...] = ()
    must_not_include: tuple[str, ...] = ()
    must_match: tuple[str, ...] = ()
    # The answer should decline: a provider refusal, or text containing one of the
    # golden set's decline markers.
    expect_decline: bool = False

    @field_validator("must_include")
    @classmethod
    def _no_empty_groups(cls, value: tuple[Phrase, ...]) -> tuple[Phrase, ...]:
        for phrase in value:
            if isinstance(phrase, tuple) and not phrase:
                raise ValueError("an alternatives group must not be empty")
        return value

    @field_validator("must_match")
    @classmethod
    def _regexes_compile(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _at_least_one(self) -> Expectations:
        if not (
            self.must_include or self.must_not_include or self.must_match or self.expect_decline
        ):
            raise ValueError("a case needs at least one expectation")
        return self


class GoldenCase(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    question: str = Field(min_length=1)
    expect: Expectations
    tags: tuple[str, ...] = ()
    notes: str = ""


class GoldenSet(_Strict):
    name: str
    version: str
    # True for hand-written examples that exercise the plumbing. Real golden sets
    # are labeled by the repo owner (Phase 1b).
    synthetic: bool
    description: str = ""
    decline_markers: tuple[str, ...] = DEFAULT_DECLINE_MARKERS
    cases: tuple[GoldenCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> GoldenSet:
        ids = [c.id for c in self.cases]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate case ids: {sorted(dupes)}")
        return self


class LoadedGoldenSet(_Strict):
    golden: GoldenSet
    path: str
    sha256: str


def load_golden_set(path: Path) -> LoadedGoldenSet:
    try:
        raw = path.read_bytes()
        golden = GoldenSet.model_validate(yaml.safe_load(raw))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigurationError(f"invalid golden set {path}: {exc}") from exc
    return LoadedGoldenSet(golden=golden, path=str(path), sha256=hashlib.sha256(raw).hexdigest())
