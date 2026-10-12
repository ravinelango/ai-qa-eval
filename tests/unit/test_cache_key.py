from __future__ import annotations

from typing import Any

import pytest

from apps.core import cache
from apps.core.cache import CachePolicy, cache_key

PAYLOAD: dict[str, Any] = {
    "model": "m-1",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    "output_config": {"effort": "medium"},
}


def _key(**overrides: Any) -> str:
    args: dict[str, Any] = {"provider": "anthropic", "resolved_model_id": "m-1", "payload": PAYLOAD}
    args.update(overrides)
    return cache_key(**args)


def test_key_is_deterministic_and_independent_of_dict_order() -> None:
    reordered = dict(reversed(list(PAYLOAD.items())))
    assert _key() == _key(payload=reordered)
    assert len(_key()) == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "other"},
        {"resolved_model_id": "m-2"},
        {"payload": {**PAYLOAD, "max_tokens": 101}},
        {"payload": {**PAYLOAD, "output_config": {"effort": "high"}}},
        {"payload": {**PAYLOAD, "system": "new prompt version"}},
    ],
)
def test_key_changes_when_any_input_changes(overrides: dict[str, Any]) -> None:
    assert _key(**overrides) != _key()


def test_key_changes_with_cache_schema_version(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _key()
    monkeypatch.setattr(cache, "CACHE_SCHEMA_VERSION", cache.CACHE_SCHEMA_VERSION + 1)
    assert _key() != before


def test_policies_are_explicit_strings_for_run_files() -> None:
    assert {p.value for p in CachePolicy} == {"use", "bypass", "refresh"}
