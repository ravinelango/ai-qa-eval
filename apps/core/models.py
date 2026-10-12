"""Model registry: loads config/models.yaml and resolves aliases.

All model resolution goes through `ModelRegistry.resolve`. Code never names a
provider model ID directly.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from apps.core.errors import ConfigurationError
from apps.core.types import CompletionRequest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_PATH = REPO_ROOT / "config" / "models.yaml"

ProviderName = Literal["anthropic", "fake"]
PAID_PROVIDERS: frozenset[str] = frozenset({"anthropic"})


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Rates(_Strict):
    """USD per million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_write_5m: float = Field(ge=0)
    cache_read: float = Field(ge=0)


class LongContextRates(Rates):
    threshold_input_tokens: int = Field(gt=0)


class Pricing(Rates):
    long_context: LongContextRates | None = None

    def rates_for(self, total_input_tokens: int) -> Rates:
        lc = self.long_context
        if lc is not None and total_input_tokens > lc.threshold_input_tokens:
            return lc
        return self


class ProviderSettings(_Strict):
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0, le=10)
    backoff_initial_s: float = Field(ge=0)
    backoff_max_s: float = Field(ge=0)


class ModelSpec(_Strict):
    alias: str
    provider: ProviderName
    model_id: str = Field(min_length=1)
    supports_temperature: bool
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None
    # Optional in the schema so a model can be listed before it is priced;
    # paid execution refuses to run without it (apps.core.budget).
    pricing: Pricing | None

    @property
    def is_paid(self) -> bool:
        return self.provider in PAID_PROVIDERS

    def validate_request(self, request: CompletionRequest) -> None:
        """Refuse requests this model cannot accept, instead of silently altering them."""
        if request.temperature is not None and not self.supports_temperature:
            raise ConfigurationError(
                f"model alias {self.alias!r} ({self.model_id}) does not accept temperature; "
                "leave CompletionRequest.temperature unset"
            )


class _ModelEntry(_Strict):
    provider: ProviderName
    model_id: str = Field(min_length=1)
    supports_temperature: bool
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    pricing: Pricing | None = None


class _ModelsFile(_Strict):
    pricing_checked: date
    pricing_source: str
    providers: dict[ProviderName, ProviderSettings]
    models: dict[str, _ModelEntry]

    @model_validator(mode="after")
    def _consistent(self) -> _ModelsFile:
        if self.pricing_checked > date.today():
            raise ValueError("pricing_checked is in the future")
        for alias, entry in self.models.items():
            if entry.provider not in self.providers:
                raise ValueError(f"model {alias!r} uses unconfigured provider {entry.provider!r}")
        seen: dict[tuple[str, str], str] = {}
        for alias, entry in self.models.items():
            key = (entry.provider, entry.model_id)
            if key in seen:
                raise ValueError(f"aliases {seen[key]!r} and {alias!r} map to the same model")
            seen[key] = alias
        return self


class ModelRegistry:
    def __init__(self, data: _ModelsFile, source_sha256: str) -> None:
        self._data = data
        self.source_sha256 = source_sha256

    @classmethod
    def load(cls, path: Path = DEFAULT_MODELS_PATH) -> ModelRegistry:
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise ConfigurationError(f"cannot read model config {path}: {exc}") from exc
        try:
            data = _ModelsFile.model_validate(yaml.safe_load(raw_bytes))
        except ValidationError as exc:
            raise ConfigurationError(f"invalid model config {path}:\n{exc}") from exc
        return cls(data, hashlib.sha256(raw_bytes).hexdigest())

    @property
    def pricing_checked(self) -> str:
        return self._data.pricing_checked.isoformat()

    @property
    def aliases(self) -> list[str]:
        return sorted(self._data.models)

    def resolve(self, alias: str) -> ModelSpec:
        entry = self._data.models.get(alias)
        if entry is None:
            known = ", ".join(self.aliases)
            raise ConfigurationError(f"unknown model alias {alias!r}; known aliases: {known}")
        return ModelSpec(alias=alias, **entry.model_dump())

    def provider_settings(self, provider: str) -> ProviderSettings:
        for name, settings in self._data.providers.items():
            if name == provider:
                return settings
        raise ConfigurationError(f"no settings for provider {provider!r}")
