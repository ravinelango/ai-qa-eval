"""The retrieval smoke check: spec rules and hit matching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from apps.core.errors import ConfigurationError
from apps.core.models import REPO_ROOT
from apps.rag.smoke import DEFAULT_SPEC, Expectation, SmokeSpec, load_spec, matches

MANIFEST = REPO_ROOT / "data" / "index" / "chunk_manifest.json"


def test_shipped_spec_fixes_top_k_and_thresholds_and_is_marked_unreviewed() -> None:
    spec, sha = load_spec(DEFAULT_SPEC)
    assert spec.top_k == 5
    assert spec.min_hit_rate == {"section_lookup": 0.8, "paraphrase": 0.8}
    assert spec.reviewed_by_human is False
    kinds = [q.kind for q in spec.queries]
    assert kinds.count("section_lookup") == 12
    assert kinds.count("paraphrase") == 16
    # Every expectation names a document and section that exist in the frozen corpus.
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else None
    if manifest is not None:
        known = {(c["doc_id"], c["locator"]) for c in manifest["chunks"]}
        for query in spec.queries:
            for e in query.expect:
                assert any(d == e.doc_id and matches(d, loc, e) for d, loc in known), (
                    f"{query.id}: {e} matches no chunk"
                )
    assert len(sha) == 64
    # Kept apart from the human-reviewed golden set.
    assert DEFAULT_SPEC.parent.name == "dev_checks"


@pytest.mark.parametrize(
    ("locator", "expected", "hit"),
    [
        ("11.10(e)", "11.10(e)", True),
        ("11.10(e)#2", "11.10(e)", True),
        ("11.50(a)", "11.50", True),
        ("11.50-intro", "11.50", True),
        ("III.C.2", "III.C", True),
        ("11.10(e)", "11.1", False),
        ("11.300(a)", "11.30", False),
        ("11.100(a)", "11.10", False),
        ("III.CA", "III.C", False),
        ("11.50-intro#2", "11.50", True),
        ("scope-of-application", "scope", False),
        ("background-and-history", "background", False),
    ],
)
def test_matching_is_by_locator_boundary(locator: str, expected: str, hit: bool) -> None:
    assert matches("doc", locator, Expectation(doc_id="doc", locator=expected)) is hit
    assert not matches("other", locator, Expectation(doc_id="doc", locator=expected))


def _spec(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(DEFAULT_SPEC.read_text(encoding="utf-8"))
    data.update(overrides)
    return data


def test_spec_rules() -> None:
    with pytest.raises(ValidationError):
        SmokeSpec.model_validate(_spec(reviewed_by_human=True))
    with pytest.raises(ValidationError, match="every query kind needs a threshold"):
        SmokeSpec.model_validate(_spec(min_hit_rate={"paraphrase": 0.8}))
    with pytest.raises(ValidationError, match="thresholds must be"):
        SmokeSpec.model_validate(_spec(min_hit_rate={"paraphrase": 0, "section_lookup": 1}))
    with pytest.raises(ConfigurationError):
        load_spec(Path("/nonexistent/spec.yaml"))
