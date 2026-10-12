from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from apps.core.types import (
    CompletionRequest,
    CompletionResponse,
    ExecutionStats,
    GenerationStats,
    Message,
    OpaqueBlock,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)


def _response(*blocks: TextBlock | ToolCall | OpaqueBlock) -> CompletionResponse:
    usage = Usage(input_tokens=1, output_tokens=1)
    return CompletionResponse(
        content=blocks,
        stop_reason="tool_use",
        raw_stop_reason="tool_use",
        provider="fake",
        model_alias="fake",
        resolved_model_id="fake-scripted-v1",
        served_model_id="fake-scripted-v1",
        request_id=None,
        generation=GenerationStats(
            usage=usage,
            latency_ms=1,
            estimated_cost_usd=0,
            pricing_checked="2026-10-12",
            generated_at=datetime(2026, 10, 12, tzinfo=UTC),
        ),
        execution=ExecutionStats(
            cache_hit=False, elapsed_ms=1, attempts=1, estimated_api_cost_usd=0
        ),
    )


def test_tool_call_only_in_assistant_message() -> None:
    call = ToolCall(id="c1", name="search", arguments={"q": "x"})
    Message(role="assistant", content=(call,))
    with pytest.raises(ValidationError, match="assistant"):
        Message(role="user", content=(call,))


def test_tool_result_only_in_user_message() -> None:
    result = ToolResult(tool_call_id="c1", content="ok")
    Message(role="user", content=(result,))
    with pytest.raises(ValidationError, match="user"):
        Message(role="assistant", content=(result,))


def test_empty_message_rejected() -> None:
    with pytest.raises(ValidationError):
        Message(role="user", content=())


def test_response_exposes_multiple_tool_calls_in_order() -> None:
    a = ToolCall(id="a", name="search_documents", arguments={"query": "x"})
    b = ToolCall(id="b", name="get_document_section", arguments={"section": "11.10(e)"})
    response = _response(TextBlock(text="checking"), a, b)
    assert response.tool_calls == [a, b]
    assert response.text == "checking"


def test_assistant_message_keeps_opaque_blocks_unchanged() -> None:
    thinking = OpaqueBlock(provider="anthropic", data={"type": "thinking", "signature": "sig"})
    call = ToolCall(id="a", name="search_documents", arguments={})
    message = _response(thinking, call).as_assistant_message()
    assert message.role == "assistant"
    assert message.content == (thinking, call)


def test_tool_results_for_one_turn_go_in_one_message() -> None:
    results = [ToolResult(tool_call_id="a", content="1"), ToolResult(tool_call_id="b", content="2")]
    message = Message.tool_results(results)
    assert message.role == "user"
    assert len(message.content) == 2


def test_request_must_start_with_user_turn() -> None:
    with pytest.raises(ValidationError, match="first message"):
        CompletionRequest(
            model="fake",
            messages=(Message(role="assistant", content=(TextBlock(text="hi"),)),),
            max_tokens=10,
        )


def test_request_rejects_non_positive_max_tokens() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(model="fake", messages=(Message.user("hi"),), max_tokens=0)


def test_tool_name_must_match_api_pattern() -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name="bad name!", description="x", input_schema={})


def test_usage_rejects_negative_counts_and_keeps_unreported_cache_fields_none() -> None:
    usage = Usage(input_tokens=3, output_tokens=4)
    assert usage.cache_read_input_tokens is None
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1, output_tokens=0)


def test_types_are_immutable() -> None:
    block = TextBlock(text="a")
    with pytest.raises(ValidationError):
        block.text = "b"  # type: ignore[misc]
