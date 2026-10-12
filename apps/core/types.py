"""Provider-neutral request/response contract.

Every provider (Anthropic, the fake, any future one) speaks these types. They are
pydantic models so they serialize deterministically into cache keys and saved
run files.

Design notes (docs/decisions/0002-provider-contract.md):
- Tool calls are structured (id, name, arguments). One response may carry several.
- Tool results travel back in a user message, keyed by the tool call id.
- `OpaqueBlock` carries provider-specific content (e.g. Anthropic thinking blocks)
  that must be echoed back unchanged in multi-turn tool loops.
- Usage is never invented: a successful response must report token counts, and a
  failed call carries no usage at all (see apps.core.errors).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextBlock(_Frozen):
    type: Literal["text"] = "text"
    text: str


class ToolCall(_Frozen):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(_Frozen):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str
    is_error: bool = False


class OpaqueBlock(_Frozen):
    """Provider-specific content that callers must round-trip without editing."""

    type: Literal["opaque"] = "opaque"
    provider: str
    data: dict[str, Any]


ContentBlock = Annotated[
    TextBlock | ToolCall | ToolResult | OpaqueBlock, Field(discriminator="type")
]
ResponseBlock = Annotated[TextBlock | ToolCall | OpaqueBlock, Field(discriminator="type")]


class Message(_Frozen):
    role: Literal["user", "assistant"]
    content: tuple[ContentBlock, ...]

    @model_validator(mode="after")
    def _blocks_match_role(self) -> Message:
        if not self.content:
            raise ValueError("message content must not be empty")
        for block in self.content:
            if isinstance(block, ToolCall) and self.role != "assistant":
                raise ValueError("tool calls can only appear in assistant messages")
            if isinstance(block, ToolResult) and self.role != "user":
                raise ValueError("tool results can only appear in user messages")
        return self

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=(TextBlock(text=text),))

    @classmethod
    def tool_results(cls, results: list[ToolResult]) -> Message:
        """All results for one assistant turn go back in a single user message."""
        return cls(role="user", content=tuple(results))


class ToolSpec(_Frozen):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str
    input_schema: dict[str, Any]


class CompletionRequest(_Frozen):
    """One model call. `model` is an alias from config/models.yaml, never a raw ID."""

    model: str
    messages: tuple[Message, ...]
    max_tokens: int = Field(gt=0)
    system: str | None = None
    tools: tuple[ToolSpec, ...] = ()
    # None means "provider default". Models that reject sampling parameters raise
    # ConfigurationError if this is set, rather than silently dropping it.
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    stop_sequences: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _conversation_shape(self) -> CompletionRequest:
        if not self.messages:
            raise ValueError("messages must not be empty")
        if self.messages[0].role != "user":
            raise ValueError("the first message must come from the user")
        return self


StopReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal", "pause_turn", "unknown"
]


class Usage(_Frozen):
    """Token counts reported by the provider for one successful attempt.

    Cache fields are None when the provider did not report them; they are never
    filled with an assumed zero.
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)


class GenerationStats(_Frozen):
    """Facts about the API call that produced this content.

    For a cache hit these describe the original call, not the current one.
    """

    usage: Usage
    latency_ms: float = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    pricing_checked: str
    generated_at: datetime


class ExecutionStats(_Frozen):
    """Facts about this execution of the call (cache hit or fresh request)."""

    cache_hit: bool
    elapsed_ms: float = Field(ge=0)
    attempts: int = Field(ge=0)
    # Estimated spend caused by *this* execution, from reported usage: 0 for a
    # cache hit. Attempts that failed with an error status add nothing.
    estimated_api_cost_usd: float = Field(ge=0)
    # Worst-case estimate for attempts that may have been billed without reporting
    # usage (timeouts, dropped connections). Kept separate, never treated as free.
    uncertain_cost_usd: float = Field(default=0.0, ge=0)


class CompletionResponse(_Frozen):
    content: tuple[ResponseBlock, ...]
    stop_reason: StopReason
    raw_stop_reason: str | None
    refusal_category: str | None = None
    provider: str
    model_alias: str
    resolved_model_id: str
    served_model_id: str
    request_id: str | None
    generation: GenerationStats
    execution: ExecutionStats

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.content if isinstance(b, ToolCall)]

    def as_assistant_message(self) -> Message:
        """The assistant turn to append to history, opaque blocks included."""
        return Message(role="assistant", content=tuple(self.content))
