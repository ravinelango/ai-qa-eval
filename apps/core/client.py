"""ModelClient: the one entry point application code uses to call a model.

Per call it:
1. resolves the alias through ModelRegistry (no raw model IDs in callers),
2. refuses paid providers unless paid execution was explicitly enabled and a
   BudgetGuard is attached,
3. validates the request against the model before spending anything,
4. runs attempts with our own retry policy; each attempt reserves budget first,
5. records one "generation" observation with inputs, outputs, usage, cost,
   attempt errors and the final error if the call fails.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol

from apps.core.budget import BudgetGuard
from apps.core.errors import (
    BudgetConfigError,
    ConfigurationError,
    PaidCallsNotAllowedError,
    ProviderError,
)
from apps.core.models import ModelRegistry, ModelSpec, ProviderSettings
from apps.core.pricing import cost_of_usage, require_pricing
from apps.core.providers.base import AttemptResult, ProviderBackend
from apps.core.tracing.base import Tracer
from apps.core.tracing.local import NoopTracer
from apps.core.types import (
    CompletionRequest,
    CompletionResponse,
    ExecutionStats,
    FailedAttempt,
    GenerationStats,
)

MAX_BACKOFF_CAP_S = 60.0

logger = logging.getLogger(__name__)


class Completer(Protocol):
    """What application code depends on. ModelClient implements it; tests may fake it."""

    def complete(self, request: CompletionRequest) -> CompletionResponse: ...


def backoff_delay(settings: ProviderSettings, attempt: int, retry_after_s: float | None) -> float:
    """Delay before retry number `attempt` (1-based). Honors retry-after when given."""
    if retry_after_s is not None:
        return min(max(retry_after_s, 0.0), MAX_BACKOFF_CAP_S)
    exponential = settings.backoff_initial_s * (2.0 ** (attempt - 1))
    return min(exponential, settings.backoff_max_s)


class ModelClient:
    def __init__(
        self,
        *,
        registry: ModelRegistry,
        backends: Mapping[str, ProviderBackend],
        tracer: Tracer | None = None,
        allow_paid: bool = False,
        guard: BudgetGuard | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.registry = registry
        self._backends = dict(backends)
        self.tracer = tracer or NoopTracer()
        self.allow_paid = allow_paid
        self.guard = guard
        self._sleep = sleep
        self._clock = clock
        self._now = now

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        spec = self.registry.resolve(request.model)
        self._check_paid_allowed(spec)
        backend = self._backends.get(spec.provider)
        if backend is None:
            raise ConfigurationError(f"no backend registered for provider {spec.provider!r}")
        # Validate (e.g. unsupported temperature) before any budget is reserved.
        spec.validate_request(request)
        backend.normalized_payload(spec, request)
        settings = self.registry.provider_settings(spec.provider)

        with self.tracer.observe(
            "llm.complete",
            kind="generation",
            input=request.model_dump(mode="json", exclude={"model"}),
            model=spec.model_id,
            model_parameters={
                "max_tokens": request.max_tokens,
                "effort": spec.effort,
                "temperature": request.temperature,
            },
            metadata={"model_alias": spec.alias, "provider": spec.provider},
        ) as observation:
            history: list[FailedAttempt] = []
            try:
                response = self._run_attempts(spec, backend, settings, request, history)
            except BaseException as exc:
                # Attach the attempt history to whatever escapes (our errors declare
                # the attribute; others get it set dynamically) for saved evidence.
                setattr(exc, "failed_attempts", tuple(history))  # noqa: B010
                raise
            finally:
                if history:
                    observation.update(
                        metadata={"failed_attempts": [a.model_dump() for a in history]}
                    )
            observation.update(
                output={
                    "content": [b.model_dump(mode="json") for b in response.content],
                    "stop_reason": response.stop_reason,
                },
                usage=response.generation.usage,
                cost_usd=response.generation.estimated_cost_usd,
                metadata={
                    "served_model_id": response.served_model_id,
                    "request_id": response.request_id,
                    "attempts": response.execution.attempts,
                    "uncertain_cost_usd": response.execution.uncertain_cost_usd,
                },
            )
        return response

    def _check_paid_allowed(self, spec: ModelSpec) -> None:
        if not spec.is_paid:
            return
        if not self.allow_paid:
            raise PaidCallsNotAllowedError(
                f"model alias {spec.alias!r} calls a paid API; paid execution must be "
                "enabled explicitly (e.g. --allow-paid)"
            )
        if self.guard is None:
            raise BudgetConfigError("paid execution requires a BudgetGuard with approved limits")
        require_pricing(spec)

    def _run_attempts(
        self,
        spec: ModelSpec,
        backend: ProviderBackend,
        settings: ProviderSettings,
        request: CompletionRequest,
        history: list[FailedAttempt],
    ) -> CompletionResponse:
        """Every dispatched attempt ends settled: success, error, or unknown outcome."""
        started = self._clock()
        attempts = 0
        uncertain_usd = 0.0
        while True:
            reservation = self.guard.reserve(spec, request) if spec.is_paid and self.guard else None
            attempts += 1
            try:
                result = backend.send(spec, request)
            except ProviderError as err:
                uncertain = reservation.settle_error(err) if reservation is not None else 0.0
                uncertain_usd += uncertain
                err.attempts = attempts
                history.append(err.to_failed_attempt(attempts, uncertain))
                will_retry = err.retryable and attempts <= settings.max_retries
                logger.warning(
                    "model call attempt %d failed (%s, %s): %s",
                    attempts,
                    type(err).__name__,
                    "retrying" if will_retry else "giving up",
                    err,
                )
                if not will_retry:
                    raise
                self._sleep(backoff_delay(settings, attempts, err.retry_after_s))
                continue
            except BaseException as exc:
                # Not a provider outcome we understand (a bug, an interrupt). The
                # request may have been sent, so its reservation stays as uncertain.
                uncertain = reservation.settle_unknown() if reservation is not None else 0.0
                history.append(
                    FailedAttempt(
                        attempt=attempts,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        retryable=False,
                        charge="uncertain",
                        uncertain_cost_usd=uncertain,
                    )
                )
                logger.error("model call attempt %d raised %s", attempts, type(exc).__name__)
                raise
            if reservation is not None:
                cost = reservation.settle_success(result.usage)
            else:
                cost = cost_of_usage(spec.pricing, result.usage) if spec.pricing else 0.0
            return self._build_response(
                spec, result, cost, started, attempts, uncertain_usd, tuple(history)
            )

    def _build_response(
        self,
        spec: ModelSpec,
        result: AttemptResult,
        cost: float,
        started: float,
        attempts: int,
        uncertain_usd: float,
        failed_attempts: tuple[FailedAttempt, ...],
    ) -> CompletionResponse:
        return CompletionResponse(
            content=result.content,
            stop_reason=result.stop_reason,
            raw_stop_reason=result.raw_stop_reason,
            refusal_category=result.refusal_category,
            provider=spec.provider,
            model_alias=spec.alias,
            resolved_model_id=spec.model_id,
            served_model_id=result.served_model_id,
            request_id=result.request_id,
            generation=GenerationStats(
                usage=result.usage,
                latency_ms=result.latency_ms,
                estimated_cost_usd=cost,
                pricing_checked=self.registry.pricing_checked,
                generated_at=self._now(),
            ),
            execution=ExecutionStats(
                cache_hit=False,
                elapsed_ms=(self._clock() - started) * 1000.0,
                attempts=attempts,
                estimated_api_cost_usd=cost,
                uncertain_cost_usd=uncertain_usd,
                failed_attempts=failed_attempts,
            ),
        )
