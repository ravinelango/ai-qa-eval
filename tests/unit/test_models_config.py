from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from apps.core.errors import ConfigurationError
from apps.core.models import ModelRegistry

WriteModels = Callable[[dict[str, Any]], Path]


def test_shipped_config_loads_and_resolves_every_alias(registry: ModelRegistry) -> None:
    for alias in registry.aliases:
        spec = registry.resolve(alias)
        assert spec.alias == alias
        assert spec.model_id
        registry.provider_settings(spec.provider)


def test_shipped_paid_models_have_pricing(registry: ModelRegistry) -> None:
    for alias in registry.aliases:
        spec = registry.resolve(alias)
        if spec.is_paid:
            assert spec.pricing is not None, alias


def test_unknown_alias_lists_known_aliases(registry: ModelRegistry) -> None:
    with pytest.raises(ConfigurationError, match=r"known aliases: .*fake"):
        registry.resolve("gpt-something")


def test_fake_alias_is_not_paid(registry: ModelRegistry) -> None:
    assert not registry.resolve("fake").is_paid


def test_two_aliases_for_one_model_rejected(
    models_yaml: dict[str, Any], write_models: WriteModels
) -> None:
    data = copy.deepcopy(models_yaml)
    data["models"]["opus_copy"] = copy.deepcopy(data["models"]["opus"])
    with pytest.raises(ConfigurationError, match="same model"):
        ModelRegistry.load(write_models(data))


def test_model_with_unconfigured_provider_rejected(
    models_yaml: dict[str, Any], write_models: WriteModels
) -> None:
    data = copy.deepcopy(models_yaml)
    del data["providers"]["anthropic"]
    with pytest.raises(ConfigurationError, match="unconfigured provider"):
        ModelRegistry.load(write_models(data))


def test_future_pricing_date_rejected(
    models_yaml: dict[str, Any], write_models: WriteModels
) -> None:
    data = copy.deepcopy(models_yaml)
    data["pricing_checked"] = date.today() + timedelta(days=1)
    with pytest.raises(ConfigurationError, match="future"):
        ModelRegistry.load(write_models(data))


def test_unknown_keys_rejected(models_yaml: dict[str, Any], write_models: WriteModels) -> None:
    data = copy.deepcopy(models_yaml)
    data["models"]["opus"]["temprature"] = 0
    with pytest.raises(ConfigurationError):
        ModelRegistry.load(write_models(data))


def test_model_without_pricing_loads(
    models_yaml: dict[str, Any], write_models: WriteModels
) -> None:
    """Pricing may be absent in config; spending code refuses to run without it."""
    data = copy.deepcopy(models_yaml)
    del data["models"]["opus"]["pricing"]
    assert ModelRegistry.load(write_models(data)).resolve("opus").pricing is None


def test_config_hash_changes_with_content(
    models_yaml: dict[str, Any], write_models: WriteModels, registry: ModelRegistry
) -> None:
    data = copy.deepcopy(models_yaml)
    data["models"]["opus"]["effort"] = "high"
    assert ModelRegistry.load(write_models(data)).source_sha256 != registry.source_sha256


def test_missing_file_is_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        ModelRegistry.load(tmp_path / "nope.yaml")
