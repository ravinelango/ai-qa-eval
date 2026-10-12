"""LangfuseTracer against the real Langfuse SDK, exporting spans to memory.

The Langfuse client is given its own OTel TracerProvider and an in-memory span
exporter, so spans never leave the process. Scores go through the SDK's HTTP
ingestion, so they are checked against a stand-in client instead.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from langfuse import Langfuse
from langfuse import LangfuseOtelSpanAttributes as A
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from apps.core.tracing.base import TraceAttributes
from apps.core.tracing.langfuse_tracer import LangfuseTracer
from apps.core.types import Usage


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> Iterator[LangfuseTracer]:
    # The SDK caches one client per public key, so each test uses a fresh key.
    client = Langfuse(
        public_key=f"pk-lf-test-{uuid.uuid4().hex}",
        secret_key="sk-lf-test-offline",
        base_url="http://localhost:9",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
        flush_at=1,
    )
    yield LangfuseTracer(client)
    client.flush()


def _spans(tracer: LangfuseTracer, exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    tracer.flush()
    return {span.name: span for span in exporter.get_finished_spans()}


def _attr(span: ReadableSpan, key: str) -> Any:
    assert span.attributes is not None
    return span.attributes.get(key)


def test_root_uses_our_trace_id_and_children_share_it(
    tracer: LangfuseTracer, exporter: InMemorySpanExporter
) -> None:
    with tracer.observe("rag.answer", input={"q": "x"}) as root:
        assert tracer.current_trace_id() == root.trace_id
        with tracer.observe("retrieve", kind="retriever") as child:
            child.update(output={"chunk_ids": ["c1"]})
    spans = _spans(tracer, exporter)
    root_ctx, child_ctx = spans["rag.answer"].context, spans["retrieve"].context
    assert root_ctx is not None
    assert child_ctx is not None
    assert format(root_ctx.trace_id, "032x") == root.trace_id
    assert child_ctx.trace_id == root_ctx.trace_id
    assert _attr(spans["retrieve"], A.OBSERVATION_TYPE) == "retriever"
    assert json.loads(_attr(spans["retrieve"], A.OBSERVATION_OUTPUT)) == {"chunk_ids": ["c1"]}
    assert tracer.current_trace_id() is None


def test_generation_records_model_usage_and_cost(
    tracer: LangfuseTracer, exporter: InMemorySpanExporter
) -> None:
    with tracer.observe(
        "llm.complete",
        kind="generation",
        model="claude-opus-5-5",
        model_parameters={"max_tokens": 100, "effort": "medium"},
    ) as obs:
        obs.update(
            output={"text": "hi"},
            usage=Usage(input_tokens=10, output_tokens=5, cache_read_input_tokens=2),
            cost_usd=0.0123,
        )
    span = _spans(tracer, exporter)["llm.complete"]
    assert _attr(span, A.OBSERVATION_TYPE) == "generation"
    assert _attr(span, A.OBSERVATION_MODEL) == "claude-opus-5-5"
    assert json.loads(_attr(span, A.OBSERVATION_USAGE_DETAILS)) == {
        "input": 10,
        "output": 5,
        "cache_read_input_tokens": 2,
    }
    assert json.loads(_attr(span, A.OBSERVATION_COST_DETAILS)) == {"total": 0.0123}


def test_exception_marks_observation_as_error(
    tracer: LangfuseTracer, exporter: InMemorySpanExporter
) -> None:
    with pytest.raises(RuntimeError, match="boom"), tracer.observe("failing"):
        raise RuntimeError("boom")
    span = _spans(tracer, exporter)["failing"]
    assert _attr(span, A.OBSERVATION_LEVEL) == "ERROR"
    assert _attr(span, A.OBSERVATION_STATUS_MESSAGE) == "RuntimeError: boom"


def test_trace_attributes_propagate_prompt_version_and_tags(
    tracer: LangfuseTracer, exporter: InMemorySpanExporter
) -> None:
    attributes = TraceAttributes(prompt_version="rag-v3", run_id="run-1", tags=("smoke",))
    with tracer.observe("root", attributes=attributes), tracer.observe("child"):
        pass
    child = _spans(tracer, exporter)["child"]
    assert _attr(child, A.VERSION) == "rag-v3"
    assert list(_attr(child, A.TRACE_TAGS)) == ["smoke"]


class _ScoreClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create_score(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def test_score_maps_to_create_score() -> None:
    client = _ScoreClient()
    LangfuseTracer(client).score(  # type: ignore[arg-type]
        trace_id="a" * 32, name="faithfulness", value=0.8, comment="judge v1"
    )
    assert client.calls == [
        {
            "name": "faithfulness",
            "value": 0.8,
            "trace_id": "a" * 32,
            "observation_id": None,
            "comment": "judge v1",
        }
    ]
