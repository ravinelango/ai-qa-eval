"""Scripted fake backend for tests and the offline demo.

It returns pre-written outputs in order and records every request it receives.
Nothing it returns is a model output; results produced with it are synthetic.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from apps.core.errors import ProviderError
from apps.core.models import ModelSpec
from apps.core.providers.base import AttemptResult
from apps.core.types import (
    CompletionRequest,
    ResponseBlock,
    StopReason,
    TextBlock,
    ToolCall,
    Usage,
)


@dataclass(frozen=True)
class ScriptedReply:
    content: tuple[ResponseBlock, ...]
    stop_reason: StopReason = "end_turn"
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=10, output_tokens=5))
    latency_ms: float = 1.0
    refusal_category: str | None = None

    @classmethod
    def text(cls, text: str, **kwargs: Any) -> ScriptedReply:
        return cls(content=(TextBlock(text=text),), **kwargs)

    @classmethod
    def tool_calls(cls, *calls: ToolCall, **kwargs: Any) -> ScriptedReply:
        return cls(content=tuple(calls), stop_reason="tool_use", **kwargs)


Step = ScriptedReply | ProviderError


class FakeBackend:
    """Plays back a script of replies or errors, one per attempt."""

    def __init__(self, script: Iterable[Step], *, name: str = "fake") -> None:
        self.name = name
        self._script: deque[Step] = deque(script)
        self.requests: list[CompletionRequest] = []

    def normalized_payload(self, spec: ModelSpec, request: CompletionRequest) -> dict[str, Any]:
        spec.validate_request(request)
        payload = request.model_dump(mode="json")
        payload["model"] = spec.model_id
        return payload

    def send(self, spec: ModelSpec, request: CompletionRequest) -> AttemptResult:
        self.normalized_payload(spec, request)
        self.requests.append(request)
        if not self._script:
            raise AssertionError(
                "FakeBackend script exhausted: the test made more calls than scripted"
            )
        step = self._script.popleft()
        if isinstance(step, ProviderError):
            raise step
        return AttemptResult(
            content=step.content,
            stop_reason=step.stop_reason,
            raw_stop_reason=step.stop_reason,
            refusal_category=step.refusal_category,
            usage=step.usage,
            served_model_id=spec.model_id,
            request_id=f"fake-req-{len(self.requests)}",
            latency_ms=step.latency_ms,
        )

    @property
    def remaining(self) -> int:
        return len(self._script)
