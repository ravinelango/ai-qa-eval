"""The single-attempt backend interface that each provider implements.

Backends make exactly one request per `send` call, with no retries of their own.
Retries, budget checks, timing aggregation and tracing live in
apps.core.client.ModelClient, so every attempt passes through the budget guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from apps.core.models import ModelSpec
from apps.core.types import CompletionRequest, ResponseBlock, StopReason, Usage


@dataclass(frozen=True)
class AttemptResult:
    content: tuple[ResponseBlock, ...]
    stop_reason: StopReason
    raw_stop_reason: str | None
    refusal_category: str | None
    usage: Usage
    served_model_id: str
    request_id: str | None
    latency_ms: float


class ProviderBackend(Protocol):
    name: str

    def normalized_payload(self, spec: ModelSpec, request: CompletionRequest) -> dict[str, Any]:
        """The exact request body this backend would send. Also the cache-key input.

        Raises ConfigurationError for requests the model cannot accept.
        """
        ...

    def send(self, spec: ModelSpec, request: CompletionRequest) -> AttemptResult:
        """One attempt. Raises apps.core.errors.ProviderError on failure."""
        ...
