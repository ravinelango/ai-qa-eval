"""Small builders shared by tests. All data here is synthetic test fixture data."""

from __future__ import annotations

from apps.core.types import CompletionRequest, Message


def simple_request(
    model: str = "fake", *, max_tokens: int = 100, text: str = "hi"
) -> CompletionRequest:
    return CompletionRequest(model=model, messages=(Message.user(text),), max_tokens=max_tokens)


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
