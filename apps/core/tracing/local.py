"""Tracers that need no external service: a no-op and a local recorder."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apps.core.redact import redact
from apps.core.tracing.base import (
    ObservationKind,
    ObservationRecord,
    ScoreRecord,
    TraceAttributes,
    Tracer,
    error_message,
    new_observation_id,
    new_trace_id,
)
from apps.core.types import Usage


@dataclass
class _NoopObservation:
    trace_id: str | None = None
    observation_id: str | None = None

    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage: Usage | None = None,
        cost_usd: float | None = None,
    ) -> None:
        return None

    def record_error(self, message: str) -> None:
        return None


class NoopTracer(Tracer):
    """Records nothing. Observations have no trace ID, so none is ever claimed."""

    backend = "none"

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
    ) -> Iterator[_NoopObservation]:
        yield _NoopObservation()

    def score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float | str,
        comment: str | None = None,
        observation_id: str | None = None,
    ) -> None:
        return None

    def current_trace_id(self) -> str | None:
        return None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class _RecordedObservation:
    def __init__(self, record: ObservationRecord) -> None:
        self.record = record

    @property
    def trace_id(self) -> str | None:
        return self.record.trace_id

    @property
    def observation_id(self) -> str | None:
        return self.record.observation_id

    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage: Usage | None = None,
        cost_usd: float | None = None,
    ) -> None:
        if output is not None:
            self.record.output = output
        if metadata:
            self.record.metadata.update(metadata)
        if usage is not None:
            self.record.usage = usage.model_dump()
        if cost_usd is not None:
            self.record.cost_usd = cost_usd

    def record_error(self, message: str) -> None:
        self.record.level = "ERROR"
        self.record.status_message = message


class RecordingTracer(Tracer):
    """Keeps observations in memory; optionally appends them to a JSONL file on flush.

    Used by tests (inspect `.records`) and as the `local` backend, which leaves a
    trace file next to run outputs without any hosted service.
    """

    backend = "local"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[ObservationRecord] = []
        self.scores: list[ScoreRecord] = []
        self._flushed_records = 0
        self._flushed_scores = 0
        self._current: ContextVar[ObservationRecord | None] = ContextVar(
            f"recording_tracer_{id(self)}", default=None
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
    ) -> Iterator[_RecordedObservation]:
        parent = self._current.get()
        record = ObservationRecord(
            trace_id=parent.trace_id if parent else new_trace_id(),
            observation_id=new_observation_id(),
            parent_observation_id=parent.observation_id if parent else None,
            name=name,
            kind=kind,
            started_at=_now(),
            input=input,
            metadata=dict(metadata or {}),
            model=model,
            model_parameters=model_parameters,
            attributes=attributes.as_dict() if attributes else None,
        )
        handle = _RecordedObservation(record)
        token = self._current.set(record)
        try:
            yield handle
        except BaseException as exc:
            if record.level != "ERROR":
                handle.record_error(error_message(exc))
            raise
        finally:
            self._current.reset(token)
            record.ended_at = _now()
            self.records.append(record)

    def score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float | str,
        comment: str | None = None,
        observation_id: str | None = None,
    ) -> None:
        self.scores.append(ScoreRecord(trace_id, name, value, comment, observation_id))

    def current_trace_id(self) -> str | None:
        current = self._current.get()
        return current.trace_id if current else None

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for record in self.records[self._flushed_records :]:
                line = json.dumps({"type": "observation", **asdict(record)}, default=str)
                fh.write(redact(line) + "\n")
            for score in self.scores[self._flushed_scores :]:
                fh.write(redact(json.dumps({"type": "score", **asdict(score)}, default=str)) + "\n")
        self._flushed_records = len(self.records)
        self._flushed_scores = len(self.scores)
