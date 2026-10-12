"""Langfuse adapter (Python SDK v4, OpenTelemetry-based).

Written against langfuse==4.17.0; API checked against the SDK source and the
Langfuse docs repository. Only this module imports langfuse, so swapping in
Arize Phoenix means writing one more Tracer subclass.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import Any

from langfuse import Langfuse, propagate_attributes

from apps.core.tracing.base import (
    ObservationKind,
    TraceAttributes,
    Tracer,
    error_message,
    new_trace_id,
)
from apps.core.types import Usage


def usage_details(usage: Usage) -> dict[str, int]:
    details = {"input": usage.input_tokens, "output": usage.output_tokens}
    if usage.cache_read_input_tokens is not None:
        details["cache_read_input_tokens"] = usage.cache_read_input_tokens
    if usage.cache_creation_input_tokens is not None:
        details["cache_creation_input_tokens"] = usage.cache_creation_input_tokens
    return details


class _LangfuseObservation:
    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.errored = False

    @property
    def trace_id(self) -> str | None:
        return str(self._wrapped.trace_id)

    @property
    def observation_id(self) -> str | None:
        return str(self._wrapped.id)

    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage: Usage | None = None,
        cost_usd: float | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if metadata:
            kwargs["metadata"] = metadata
        if usage is not None:
            kwargs["usage_details"] = usage_details(usage)
        if cost_usd is not None:
            kwargs["cost_details"] = {"total": cost_usd}
        if kwargs:
            self._wrapped.update(**kwargs)

    def record_error(self, message: str) -> None:
        self.errored = True
        self._wrapped.update(level="ERROR", status_message=message)


def _propagated(attributes: TraceAttributes) -> dict[str, Any]:
    # propagate_attributes metadata values must be short strings.
    metadata = {
        key: value
        for key, value in (("run_id", attributes.run_id), ("variant", attributes.variant))
        if value is not None
    }
    kwargs: dict[str, Any] = {}
    if attributes.prompt_version is not None:
        kwargs["version"] = attributes.prompt_version
    if attributes.tags:
        kwargs["tags"] = list(attributes.tags)
    if metadata:
        kwargs["metadata"] = metadata
    return kwargs


class LangfuseTracer(Tracer):
    backend = "langfuse"

    def __init__(self, client: Langfuse) -> None:
        self._client = client
        self._current_trace: ContextVar[str | None] = ContextVar(
            f"langfuse_tracer_{id(self)}", default=None
        )

    @contextmanager
    def observe(
        self,
        name: str,
        *,
        kind: ObservationKind = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        model_parameters: dict[str, str | int | float | bool | None] | None = None,
        attributes: TraceAttributes | None = None,
    ) -> Iterator[_LangfuseObservation]:
        parent_trace = self._current_trace.get()
        kwargs: dict[str, Any] = {"name": name, "as_type": kind}
        if parent_trace is None:
            # Root observation: use our own trace ID so saved results can cite it.
            kwargs["trace_context"] = {"trace_id": new_trace_id()}
        for key, value in (
            ("input", input),
            ("metadata", metadata),
            ("model", model),
            ("model_parameters", model_parameters),
        ):
            if value is not None:
                kwargs[key] = value
        with ExitStack() as stack:
            wrapped = stack.enter_context(self._client.start_as_current_observation(**kwargs))
            if attributes is not None:
                propagated = _propagated(attributes)
                if propagated:
                    stack.enter_context(propagate_attributes(**propagated))
            handle = _LangfuseObservation(wrapped)
            token = self._current_trace.set(handle.trace_id)
            try:
                yield handle
            except BaseException as exc:
                if not handle.errored:
                    handle.record_error(error_message(exc))
                raise
            finally:
                self._current_trace.reset(token)

    def score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float | str,
        comment: str | None = None,
        observation_id: str | None = None,
    ) -> None:
        self._client.create_score(
            name=name,
            value=value,
            trace_id=trace_id,
            observation_id=observation_id,
            comment=comment,
        )

    def current_trace_id(self) -> str | None:
        return self._current_trace.get()

    def flush(self) -> None:
        self._client.flush()
