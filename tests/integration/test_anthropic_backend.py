"""AnthropicBackend against the real `anthropic` SDK, with HTTP served by a mock transport.

No network: httpx2.MockTransport answers inside the process. Response bodies are
hand-written fixtures shaped like the Messages API, not recorded API output.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest

from apps.core.budget import BudgetGuard, BudgetLimits
from apps.core.client import ModelClient
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
from apps.core.evidence import budget_record
from apps.core.models import ModelRegistry
from apps.core.providers.anthropic import AnthropicBackend
from apps.core.types import (
    CompletionRequest,
    Message,
    OpaqueBlock,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from tests.helpers import SleepRecorder

Handler = Callable[[httpx2.Request], httpx2.Response]


class Recorder:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.handler(request)

    def body(self, index: int = -1) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.requests[index].content)
        return result


def _backend(handler: Handler) -> tuple[AnthropicBackend, Recorder]:
    recorder = Recorder(handler)
    http_client = anthropic.DefaultHttpxClient(
        transport=httpx2.MockTransport(recorder), trust_env=False
    )
    backend = AnthropicBackend(api_key="test-key", timeout_s=5.0, http_client=http_client)
    return backend, recorder


def _message(
    content: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    usage: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage
        if usage is not None
        else {
            "input_tokens": 120,
            "output_tokens": 30,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        **extra,
    }


def _ok(body: dict[str, Any]) -> Handler:
    return lambda _: httpx2.Response(200, json=body, headers={"request-id": "req_fixture_1"})


def _status(code: int, error_type: str, headers: dict[str, str] | None = None) -> Handler:
    body = {"type": "error", "error": {"type": error_type, "message": f"fixture {code}"}}
    return lambda _: httpx2.Response(code, json=body, headers=headers or {})


def _request(**kwargs: Any) -> CompletionRequest:
    defaults: dict[str, Any] = {
        "model": "opus",
        "messages": (Message.user("What does 11.10(e) cover?"),),
        "max_tokens": 512,
    }
    defaults.update(kwargs)
    return CompletionRequest(**defaults)


# --- request mapping -----------------------------------------------------------------


def test_payload_maps_contract_types_to_messages_api(registry: ModelRegistry) -> None:
    backend, recorder = _backend(_ok(_message([{"type": "text", "text": "ok"}])))
    thinking = OpaqueBlock(
        provider="anthropic", data={"type": "thinking", "thinking": "", "signature": "sig-1"}
    )
    request = _request(
        system="system prompt v1",
        tools=(
            ToolSpec(
                name="search_documents",
                description="Search",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ),
        messages=(
            Message.user("find it"),
            Message(
                role="assistant",
                content=(
                    thinking,
                    ToolCall(id="toolu_1", name="search_documents", arguments={"query": "x"}),
                ),
            ),
            Message.tool_results(
                [ToolResult(tool_call_id="toolu_1", content="result", is_error=False)]
            ),
        ),
    )
    backend.send(registry.resolve("opus"), request)

    body = recorder.body()
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 512
    assert body["system"] == "system prompt v1"
    assert body["output_config"] == {"effort": "medium"}
    assert "temperature" not in body
    assert body["tools"][0]["name"] == "search_documents"
    assistant = body["messages"][1]["content"]
    assert assistant[0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}
    assert assistant[1] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "search_documents",
        "input": {"query": "x"},
    }
    assert body["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "result",
        "is_error": False,
    }


def test_payload_is_what_gets_sent(registry: ModelRegistry) -> None:
    backend, recorder = _backend(_ok(_message([{"type": "text", "text": "ok"}])))
    spec, request = registry.resolve("opus"), _request(system="s")
    backend.send(spec, request)
    assert recorder.body() == backend.normalized_payload(spec, request)


def test_unsupported_temperature_never_reaches_the_api(registry: ModelRegistry) -> None:
    backend, recorder = _backend(_ok(_message([])))
    with pytest.raises(ConfigurationError, match="temperature"):
        backend.send(registry.resolve("opus"), _request(temperature=0.0))
    assert recorder.requests == []


def test_foreign_opaque_block_rejected(registry: ModelRegistry) -> None:
    backend, recorder = _backend(_ok(_message([])))
    request = _request(
        messages=(
            Message.user("x"),
            Message(role="assistant", content=(OpaqueBlock(provider="other", data={}),)),
            Message.user("y"),
        )
    )
    with pytest.raises(ConfigurationError, match="opaque"):
        backend.send(registry.resolve("opus"), request)
    assert recorder.requests == []


def test_empty_api_key_rejected() -> None:
    with pytest.raises(ConfigurationError):
        AnthropicBackend(api_key="", timeout_s=1.0)


# --- response parsing ----------------------------------------------------------------


def test_parses_text_multiple_tool_calls_and_thinking(registry: ModelRegistry) -> None:
    content: list[dict[str, Any]] = [
        {"type": "thinking", "thinking": "", "signature": "sig-2"},
        {"type": "text", "text": "Looking that up."},
        {"type": "tool_use", "id": "toolu_a", "name": "search_documents", "input": {"query": "a"}},
        {"type": "tool_use", "id": "toolu_b", "name": "get_document_section", "input": {"id": "b"}},
    ]
    backend, _ = _backend(_ok(_message(content, stop_reason="tool_use")))
    result = backend.send(registry.resolve("opus"), _request())

    assert isinstance(result.content[0], OpaqueBlock)
    assert result.content[0].data["signature"] == "sig-2"
    assert result.content[1] == TextBlock(text="Looking that up.")
    assert [b.id for b in result.content if isinstance(b, ToolCall)] == ["toolu_a", "toolu_b"]
    assert result.stop_reason == "tool_use"
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 30
    assert result.usage.cache_read_input_tokens == 0
    assert result.served_model_id == "claude-opus-5-5"
    assert result.request_id == "req_fixture_1"
    assert result.latency_ms >= 0


def test_refusal_keeps_category(registry: ModelRegistry) -> None:
    body = _message(
        [],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": "fixture"},
    )
    backend, _ = _backend(_ok(body))
    result = backend.send(registry.resolve("opus"), _request())
    assert result.stop_reason == "refusal"
    assert result.refusal_category == "cyber"


def test_unknown_stop_reason_preserved_raw(registry: ModelRegistry) -> None:
    backend, _ = _backend(_ok(_message([], stop_reason="some_future_reason")))
    result = backend.send(registry.resolve("opus"), _request())
    assert result.stop_reason == "unknown"
    assert result.raw_stop_reason == "some_future_reason"


def test_missing_usage_is_an_error_not_zero(registry: ModelRegistry) -> None:
    body = _message([{"type": "text", "text": "ok"}])
    del body["usage"]
    backend, _ = _backend(_ok(body))
    with pytest.raises(ProviderResponseError) as info:
        backend.send(registry.resolve("opus"), _request())
    assert info.value.charge == "uncertain"


# --- errors: one HTTP request per attempt, mapped to our taxonomy ---------------------


@pytest.mark.parametrize(
    ("code", "error_type", "expected", "retryable", "charge"),
    [
        (400, "invalid_request_error", ProviderBadRequestError, False, "none"),
        (401, "authentication_error", ProviderAuthError, False, "none"),
        (403, "permission_error", ProviderAuthError, False, "none"),
        (404, "not_found_error", ProviderBadRequestError, False, "none"),
        (429, "rate_limit_error", ProviderRateLimitError, True, "none"),
        (500, "api_error", ProviderServerError, True, "none"),
        (504, "timeout_error", ProviderTimeoutError, True, "uncertain"),
        (529, "overloaded_error", ProviderOverloadedError, True, "none"),
    ],
)
def test_status_errors_are_mapped_and_not_retried_by_the_sdk(
    registry: ModelRegistry,
    code: int,
    error_type: str,
    expected: type[ProviderError],
    retryable: bool,
    charge: str,
) -> None:
    backend, recorder = _backend(_status(code, error_type))
    with pytest.raises(expected) as info:
        backend.send(registry.resolve("opus"), _request())
    assert len(recorder.requests) == 1
    assert info.value.status_code == code
    assert info.value.retryable is retryable
    assert info.value.charge == charge


def test_rate_limit_carries_retry_after(registry: ModelRegistry) -> None:
    backend, _ = _backend(_status(429, "rate_limit_error", {"retry-after": "7"}))
    with pytest.raises(ProviderRateLimitError) as info:
        backend.send(registry.resolve("opus"), _request())
    assert info.value.retry_after_s == 7.0


def _raise(exc: Exception) -> Handler:
    def handler(_: httpx2.Request) -> httpx2.Response:
        raise exc

    return handler


def test_transport_timeout_is_uncertain_charge(registry: ModelRegistry) -> None:
    backend, recorder = _backend(_raise(httpx2.ReadTimeout("fixture timeout")))
    with pytest.raises(ProviderTimeoutError) as info:
        backend.send(registry.resolve("opus"), _request())
    assert len(recorder.requests) == 1
    assert info.value.charge == "uncertain"


def test_connection_failure_is_mapped(registry: ModelRegistry) -> None:
    backend, recorder = _backend(_raise(httpx2.ConnectError("fixture refused")))
    with pytest.raises(ProviderConnectionError):
        backend.send(registry.resolve("opus"), _request())
    assert len(recorder.requests) == 1


# --- through ModelClient: our retries, budget accounting ------------------------------


def test_client_retries_overload_then_succeeds_within_budget(registry: ModelRegistry) -> None:
    responses = iter(
        [
            httpx2.Response(
                529, json={"type": "error", "error": {"type": "overloaded_error", "message": "x"}}
            ),
            httpx2.Response(200, json=_message([{"type": "text", "text": "answer"}])),
        ]
    )
    backend, recorder = _backend(lambda _: next(responses))
    guard = BudgetGuard(BudgetLimits(max_request_attempts=3, max_estimated_cost_usd=1.0))
    sleep = SleepRecorder()
    client = ModelClient(
        registry=registry,
        backends={"anthropic": backend},
        allow_paid=True,
        guard=guard,
        sleep=sleep,
    )
    response = client.complete(_request())
    assert response.text == "answer"
    assert response.execution.attempts == 2
    assert len(recorder.requests) == 2
    assert len(sleep.calls) == 1
    snap = guard.snapshot()
    assert snap.attempts_dispatched == 2
    assert snap.settled_cost_usd == pytest.approx(response.generation.estimated_cost_usd)
    assert snap.outstanding_reservations_usd == pytest.approx(0.0)


# --- regression: review finding 3 (malformed 200 responses) ---------------------------


def _negative_usage_body() -> dict[str, Any]:
    return _message(
        [{"type": "text", "text": "ok"}], usage={"input_tokens": 10, "output_tokens": -1}
    )


def test_malformed_200_becomes_provider_response_error(registry: ModelRegistry) -> None:
    backend, recorder = _backend(_ok(_negative_usage_body()))
    with pytest.raises(ProviderResponseError) as info:
        backend.send(registry.resolve("opus"), _request())
    assert info.value.charge == "uncertain"
    assert info.value.request_id == "req_fixture_1"
    assert "malformed response body" in str(info.value)
    assert len(recorder.requests) == 1


def test_malformed_200_is_fully_accounted_through_client(registry: ModelRegistry) -> None:
    backend, _ = _backend(_ok(_negative_usage_body()))
    guard = BudgetGuard(BudgetLimits(max_request_attempts=3, max_estimated_cost_usd=1.0))
    client = ModelClient(
        registry=registry, backends={"anthropic": backend}, allow_paid=True, guard=guard
    )
    with pytest.raises(ProviderResponseError) as info:
        client.complete(_request())

    snap = guard.snapshot()
    assert snap.attempts_dispatched == 1  # not retried
    assert snap.outstanding_reservations_usd == pytest.approx(0.0)
    assert snap.uncertain_cost_usd > 0  # possibly billed, kept as uncertain
    (attempt,) = info.value.failed_attempts
    assert attempt.error_type == "ProviderResponseError"
    assert attempt.charge == "uncertain"
    assert attempt.uncertain_cost_usd == pytest.approx(snap.uncertain_cost_usd)

    record = budget_record(snap)
    assert record is not None
    assert record["outstanding_reservations_usd"] == pytest.approx(0.0)
    assert record["committed_cost_usd"] == pytest.approx(snap.uncertain_cost_usd)
