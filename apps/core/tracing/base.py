"""Backend-neutral tracing interface (docs/decisions/0005-tracing-and-evidence.md).

One primitive, `Tracer.observe`, opens an observation as a context manager. With
no observation open it starts a new trace. Observations carry inputs, outputs,
metadata, model usage/cost and errors; `kind` lets later phases record retrieval
("retriever"), tool calls ("tool"), guardrails and judges without new methods.

Trace IDs are generated here (W3C format: 32 hex chars) for every recording
backend, so the same ID appears in saved run files, local trace files and
Langfuse. Traces are supplementary evidence: saved run files must stand alone.
"""

from __future__ import annotations

import secrets
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from apps.core.redact import redact
from apps.core.types import Usage

ObservationKind = Literal[
    "span", "generation", "retriever", "tool", "agent", "evaluator", "guardrail"
]
Level = Literal["DEFAULT", "WARNING", "ERROR"]


@dataclass(frozen=True)
class TraceAttributes:
    """Attributes that apply to an observation and everything nested inside it."""

    prompt_version: str | None = None
    run_id: str | None = None
    variant: str | None = None
    tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_version": self.prompt_version,
            "run_id": self.run_id,
            "variant": self.variant,
            "tags": list(self.tags),
        }


class Observation(Protocol):
    @property
    def trace_id(self) -> str | None: ...

    @property
    def observation_id(self) -> str | None: ...

    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage: Usage | None = None,
        cost_usd: float | None = None,
    ) -> None: ...

    def record_error(self, message: str) -> None: ...


class Tracer(ABC):
    backend: str

    @abstractmethod
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
    ) -> AbstractContextManager[Observation]:
        """Open an observation. Exceptions inside the block are recorded, then re-raised."""

    @abstractmethod
    def score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float | str,
        comment: str | None = None,
        observation_id: str | None = None,
    ) -> None: ...

    @abstractmethod
    def current_trace_id(self) -> str | None: ...

    def flush(self) -> None:  # noqa: B027 - optional hook
        """Deliver buffered data. Call before a short-lived process exits."""


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_observation_id() -> str:
    return secrets.token_hex(8)


def error_message(exc: BaseException) -> str:
    return redact(f"{type(exc).__name__}: {exc}")


@dataclass
class ObservationRecord:
    """A finished observation as stored by the local recorder."""

    trace_id: str
    observation_id: str
    parent_observation_id: str | None
    name: str
    kind: ObservationKind
    started_at: str
    ended_at: str | None = None
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    model_parameters: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    level: Level = "DEFAULT"
    status_message: str | None = None
    attributes: dict[str, Any] | None = None


@dataclass
class ScoreRecord:
    trace_id: str
    name: str
    value: float | str
    comment: str | None
    observation_id: str | None
