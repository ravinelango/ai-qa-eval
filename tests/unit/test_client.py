"""ModelClient behavior, using FakeBackend in place of the Anthropic backend.

Registering the fake under the name "anthropic" makes the client treat it as a
paid provider, so budget and opt-in rules are exercised without any network.
"""

from __future__ import annotations

import pytest

from apps.core.budget import BudgetGuard, BudgetLimits
from apps.core.client import ModelClient, backoff_delay
from apps.core.errors import (
    BudgetConfigError,
    BudgetExceededError,
    ConfigurationError,
    PaidCallsNotAllowedError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from apps.core.models import ModelRegistry
from apps.core.providers.fake import FakeBackend, ScriptedReply, Step
from apps.core.tracing.local import RecordingTracer
from apps.core.types import CompletionRequest, Message, ToolCall, Usage
from tests.helpers import SleepRecorder, simple_request

DEFAULT_LIMITS = BudgetLimits(10, 10.0)
OK = ScriptedReply.text("ok", usage=Usage(input_tokens=1_000, output_tokens=100))


def _paid_client(
    registry: ModelRegistry,
    script: list[Step],
    *,
    limits: BudgetLimits | None = DEFAULT_LIMITS,
    allow_paid: bool = True,
) -> tuple[ModelClient, FakeBackend, RecordingTracer, SleepRecorder]:
    backend = FakeBackend(script, name="anthropic")
    tracer = RecordingTracer()
    sleep = SleepRecorder()
    client = ModelClient(
        registry=registry,
        backends={"anthropic": backend},
        tracer=tracer,
        allow_paid=allow_paid,
        guard=BudgetGuard(limits) if limits else None,
        sleep=sleep,
    )
    return client, backend, tracer, sleep


# --- opt-in --------------------------------------------------------------------------


def test_paid_model_refused_without_explicit_opt_in(registry: ModelRegistry) -> None:
    client, backend, _, _ = _paid_client(registry, [OK], allow_paid=False)
    with pytest.raises(PaidCallsNotAllowedError):
        client.complete(simple_request("opus"))
    assert backend.requests == []


def test_paid_model_refused_without_budget_guard(registry: ModelRegistry) -> None:
    client, backend, _, _ = _paid_client(registry, [OK], limits=None)
    with pytest.raises(BudgetConfigError):
        client.complete(simple_request("opus"))
    assert backend.requests == []


def test_unpaid_model_needs_no_opt_in(registry: ModelRegistry) -> None:
    client = ModelClient(registry=registry, backends={"fake": FakeBackend([OK])})
    assert client.complete(simple_request("fake")).text == "ok"


# --- success path --------------------------------------------------------------------


def test_success_reports_usage_cost_and_both_stat_blocks(registry: ModelRegistry) -> None:
    client, _, _, _ = _paid_client(registry, [OK])
    response = client.complete(simple_request("opus"))
    expected_cost = (1_000 * 4.0 + 100 * 20.0) / 1e6
    assert response.stop_reason == "end_turn"
    assert response.resolved_model_id == "claude-opus-5-5"
    assert response.generation.usage.input_tokens == 1_000
    assert response.generation.estimated_cost_usd == pytest.approx(expected_cost)
    assert response.generation.pricing_checked == registry.pricing_checked
    assert response.execution.cache_hit is False
    assert response.execution.attempts == 1
    assert response.execution.estimated_api_cost_usd == pytest.approx(expected_cost)
    assert response.execution.uncertain_cost_usd == 0.0


def test_multiple_tool_calls_round_trip(registry: ModelRegistry) -> None:
    calls = (
        ToolCall(id="a", name="search_documents", arguments={"query": "x"}),
        ToolCall(id="b", name="search_documents", arguments={"query": "y"}),
    )
    client, _, _, _ = _paid_client(registry, [ScriptedReply.tool_calls(*calls)])
    response = client.complete(simple_request("opus"))
    assert response.stop_reason == "tool_use"
    assert [c.id for c in response.tool_calls] == ["a", "b"]


# --- retries and errors --------------------------------------------------------------


def test_retryable_error_is_retried_and_honors_retry_after(registry: ModelRegistry) -> None:
    rate_limited = ProviderRateLimitError("429", provider="anthropic", retry_after_s=7.0)
    client, backend, _, sleep = _paid_client(registry, [rate_limited, OK])
    response = client.complete(simple_request("opus"))
    assert response.execution.attempts == 2
    assert sleep.calls == [7.0]
    assert len(backend.requests) == 2
    assert client.guard is not None
    assert client.guard.snapshot().attempts_dispatched == 2


def test_non_retryable_error_raises_immediately_without_usage(registry: ModelRegistry) -> None:
    bad = ProviderBadRequestError("400", provider="anthropic", status_code=400)
    client, backend, _, sleep = _paid_client(registry, [bad, OK])
    with pytest.raises(ProviderBadRequestError) as info:
        client.complete(simple_request("opus"))
    assert info.value.attempts == 1
    assert sleep.calls == []
    assert backend.remaining == 1
    assert "usage" not in info.value.describe()


def test_retries_stop_after_configured_maximum(registry: ModelRegistry) -> None:
    max_retries = registry.provider_settings("anthropic").max_retries
    errors: list[Step] = [
        ProviderServerError("500", provider="anthropic", status_code=500)
        for _ in range(max_retries + 2)
    ]
    client, backend, _, sleep = _paid_client(registry, errors)
    with pytest.raises(ProviderServerError) as info:
        client.complete(simple_request("opus"))
    assert info.value.attempts == max_retries + 1
    assert len(backend.requests) == max_retries + 1
    assert len(sleep.calls) == max_retries


def test_budget_exhaustion_stops_before_dispatching_a_retry(registry: ModelRegistry) -> None:
    server_error = ProviderServerError("500", provider="anthropic", status_code=500)
    client, backend, _, _ = _paid_client(registry, [server_error, OK], limits=BudgetLimits(1, 10.0))
    with pytest.raises(BudgetExceededError):
        client.complete(simple_request("opus"))
    assert len(backend.requests) == 1  # the retry was never sent


def test_timeout_then_success_reports_uncertain_cost_separately(registry: ModelRegistry) -> None:
    timeout = ProviderTimeoutError("timed out", provider="anthropic")
    client, _, _, _ = _paid_client(registry, [timeout, OK])
    response = client.complete(simple_request("opus"))
    assert response.execution.attempts == 2
    assert response.execution.uncertain_cost_usd > 0
    assert client.guard is not None
    snap = client.guard.snapshot()
    assert snap.uncertain_cost_usd == pytest.approx(response.execution.uncertain_cost_usd)
    assert snap.settled_cost_usd == pytest.approx(response.execution.estimated_api_cost_usd)


def test_unsupported_temperature_rejected_before_any_spend(registry: ModelRegistry) -> None:
    client, backend, _, _ = _paid_client(registry, [OK])
    request = CompletionRequest(
        model="opus", messages=(Message.user("hi"),), max_tokens=10, temperature=0.0
    )
    with pytest.raises(ConfigurationError, match="temperature"):
        client.complete(request)
    assert backend.requests == []
    assert client.guard is not None
    assert client.guard.snapshot().attempts_dispatched == 0


def test_missing_backend_is_configuration_error(registry: ModelRegistry) -> None:
    client = ModelClient(registry=registry, backends={})
    with pytest.raises(ConfigurationError, match="no backend"):
        client.complete(simple_request("fake"))


# --- tracing -------------------------------------------------------------------------


def test_success_is_traced_with_inputs_outputs_usage_and_cost(registry: ModelRegistry) -> None:
    client, _, tracer, _ = _paid_client(registry, [OK])
    response = client.complete(simple_request("opus", text="question?"))
    (record,) = tracer.records
    assert record.kind == "generation"
    assert record.model == "claude-opus-5-5"
    assert record.input["messages"][0]["content"][0]["text"] == "question?"
    assert record.output["content"][0]["text"] == "ok"
    assert record.usage == response.generation.usage.model_dump()
    assert record.cost_usd == pytest.approx(response.generation.estimated_cost_usd)
    assert record.level == "DEFAULT"


def test_failure_is_traced_as_error_with_every_attempt(registry: ModelRegistry) -> None:
    errors: list[Step] = [
        ProviderRateLimitError("429", provider="anthropic", retry_after_s=0.0),
        ProviderBadRequestError("400", provider="anthropic", status_code=400),
    ]
    client, _, tracer, _ = _paid_client(registry, errors)
    with pytest.raises(ProviderBadRequestError):
        client.complete(simple_request("opus"))
    (record,) = tracer.records
    assert record.level == "ERROR"
    assert record.status_message is not None
    assert "ProviderBadRequestError" in record.status_message
    assert [e["error_type"] for e in record.metadata["attempt_errors"]] == [
        "ProviderRateLimitError",
        "ProviderBadRequestError",
    ]
    assert record.usage is None
    assert record.cost_usd is None


# --- backoff ------------------------------------------------------------------------


def test_backoff_is_exponential_and_capped(registry: ModelRegistry) -> None:
    settings = registry.provider_settings("anthropic")
    delays = [backoff_delay(settings, n, None) for n in range(1, 8)]
    assert delays[0] == settings.backoff_initial_s
    assert delays[1] == settings.backoff_initial_s * 2
    assert max(delays) == settings.backoff_max_s
    assert backoff_delay(settings, 1, 1_000.0) == 60.0
