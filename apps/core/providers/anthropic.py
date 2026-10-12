"""Anthropic Messages API backend (official `anthropic` SDK).

One attempt per `send`: the SDK client is built with max_retries=0 so retries
happen in ModelClient, where each attempt is budget-checked.

Notes from the current API docs (docs/decisions/0002-provider-contract.md):
- Current models reject sampling parameters, so `temperature` is sent only for
  models whose config says they support it; otherwise the request is refused
  locally instead of being silently altered.
- Effort is pinned per model in config/models.yaml and sent as output_config.
- Thinking blocks come back as OpaqueBlock and must be echoed back unchanged in
  tool loops; they are passed through verbatim.
- Server-side refusal fallbacks are deliberately not enabled: a fallback would
  change which model answered, which would corrupt per-model measurements.
  Refusals are surfaced as stop_reason "refusal".
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import anthropic
import httpx2

from apps.core.errors import (
    ConfigurationError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderConnectionError,
    ProviderError,
    ProviderOverloadedError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from apps.core.models import ModelSpec
from apps.core.providers.base import AttemptResult
from apps.core.types import (
    CompletionRequest,
    ContentBlock,
    OpaqueBlock,
    ResponseBlock,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
)

PROVIDER = "anthropic"
_KNOWN_STOP_REASONS: frozenset[str] = frozenset(
    {"end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal", "pause_turn"}
)


class AnthropicBackend:
    name = PROVIDER

    def __init__(
        self,
        *,
        api_key: str,
        timeout_s: float,
        http_client: httpx2.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_s,
            max_retries=0,
            http_client=http_client,
        )
        self._clock = clock

    # -- request ------------------------------------------------------------------

    def normalized_payload(self, spec: ModelSpec, request: CompletionRequest) -> dict[str, Any]:
        if spec.provider != PROVIDER:
            raise ConfigurationError(f"alias {spec.alias!r} is not an Anthropic model")
        spec.validate_request(request)
        payload: dict[str, Any] = {
            "model": spec.model_id,
            "max_tokens": request.max_tokens,
            "messages": [
                {"role": m.role, "content": [_block_to_api(b) for b in m.content]}
                for m in request.messages
            ],
        }
        if request.system is not None:
            payload["system"] = request.system
        if request.tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in request.tools
            ]
        if spec.effort is not None:
            payload["output_config"] = {"effort": spec.effort}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop_sequences:
            payload["stop_sequences"] = list(request.stop_sequences)
        return payload

    # -- one attempt --------------------------------------------------------------

    def send(self, spec: ModelSpec, request: CompletionRequest) -> AttemptResult:
        payload = self.normalized_payload(spec, request)
        started = self._clock()
        try:
            message = self._client.messages.create(**payload)
        except anthropic.AnthropicError as exc:
            raise _translate_error(exc) from exc
        latency_ms = (self._clock() - started) * 1000.0
        try:
            return _parse_message(message, latency_ms)
        except ProviderError:
            raise
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            # A 200 whose body breaks our contract (e.g. negative token counts). The
            # request was processed, so the charge is uncertain, not zero.
            # (pydantic's ValidationError is a ValueError.)
            raise ProviderResponseError(
                f"malformed response body: {type(exc).__name__}: {exc}",
                provider=PROVIDER,
                request_id=getattr(message, "_request_id", None),
            ) from exc


def _block_to_api(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolCall):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.arguments}
    if isinstance(block, ToolResult):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_call_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    if block.provider != PROVIDER:
        raise ConfigurationError(
            f"cannot send an opaque block from provider {block.provider!r} to Anthropic"
        )
    return dict(block.data)


def _parse_message(message: Any, latency_ms: float) -> AttemptResult:
    request_id = getattr(message, "_request_id", None)
    usage = getattr(message, "usage", None)
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        raise ProviderResponseError(
            "response did not report token usage", provider=PROVIDER, request_id=request_id
        )
    content: list[ResponseBlock] = []
    for block in message.content:
        if block.type == "text":
            content.append(TextBlock(text=block.text))
        elif block.type == "tool_use":
            content.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        else:
            # thinking, redacted_thinking and any block type we do not model
            content.append(OpaqueBlock(provider=PROVIDER, data=block.to_dict()))
    raw_stop = message.stop_reason
    stop_reason = cast(StopReason, raw_stop if raw_stop in _KNOWN_STOP_REASONS else "unknown")
    refusal_category = None
    stop_details = getattr(message, "stop_details", None)
    if raw_stop == "refusal" and stop_details is not None:
        refusal_category = getattr(stop_details, "category", None)
    return AttemptResult(
        content=tuple(content),
        stop_reason=stop_reason,
        raw_stop_reason=raw_stop,
        refusal_category=refusal_category,
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
        ),
        served_model_id=message.model,
        request_id=request_id,
        latency_ms=latency_ms,
    )


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    value = exc.response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _translate_error(exc: anthropic.AnthropicError) -> ProviderError:
    """Map SDK exceptions to our taxonomy, most specific first."""
    common: dict[str, Any] = {"provider": PROVIDER}
    if isinstance(exc, anthropic.APITimeoutError):
        return ProviderTimeoutError(f"request timed out: {exc}", **common)
    if isinstance(exc, anthropic.APIConnectionError):
        return ProviderConnectionError(f"connection failed: {exc}", **common)
    if isinstance(exc, anthropic.APIStatusError):
        status_kwargs: dict[str, Any] = {
            **common,
            "status_code": exc.status_code,
            "request_id": exc.request_id,
            "retry_after_s": _retry_after(exc),
        }
        message = f"HTTP {exc.status_code}: {exc.message}"
        if isinstance(exc, anthropic.RateLimitError):
            return ProviderRateLimitError(message, **status_kwargs)
        if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            return ProviderAuthError(message, **status_kwargs)
        if exc.status_code == 504:
            # Gateway timeout: the request may have been processed upstream.
            return ProviderTimeoutError(message, **status_kwargs)
        if exc.status_code == 529:
            return ProviderOverloadedError(message, **status_kwargs)
        if exc.status_code >= 500:
            return ProviderServerError(message, **status_kwargs)
        return ProviderBadRequestError(message, **status_kwargs)
    if isinstance(exc, anthropic.APIResponseValidationError):
        return ProviderResponseError(f"unexpected response body: {exc}", **common)
    return ProviderBadRequestError(f"{type(exc).__name__}: {exc}", **common)
