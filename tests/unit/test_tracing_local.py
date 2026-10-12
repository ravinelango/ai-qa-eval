from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.core.tracing.base import TraceAttributes
from apps.core.tracing.local import NoopTracer, RecordingTracer
from apps.core.types import Usage


def test_nested_observations_share_trace_and_link_parents() -> None:
    tracer = RecordingTracer()
    with tracer.observe("root", attributes=TraceAttributes(prompt_version="v2")) as root:
        assert tracer.current_trace_id() == root.trace_id
        with tracer.observe("retrieve", kind="retriever", input={"query": "q"}) as child:
            child.update(output={"chunk_ids": ["c1", "c2"]})
    assert tracer.current_trace_id() is None
    by_name = {r.name: r for r in tracer.records}
    assert by_name["retrieve"].trace_id == by_name["root"].trace_id
    assert by_name["retrieve"].parent_observation_id == by_name["root"].observation_id
    assert by_name["retrieve"].output == {"chunk_ids": ["c1", "c2"]}
    assert by_name["root"].attributes is not None
    assert by_name["root"].attributes["prompt_version"] == "v2"
    assert len(by_name["root"].trace_id) == 32


def test_separate_roots_get_separate_traces() -> None:
    tracer = RecordingTracer()
    with tracer.observe("a"):
        pass
    with tracer.observe("b"):
        pass
    assert tracer.records[0].trace_id != tracer.records[1].trace_id


def test_exception_is_recorded_and_reraised() -> None:
    tracer = RecordingTracer()
    with pytest.raises(ValueError, match="boom"), tracer.observe("step"):
        raise ValueError("boom")
    (record,) = tracer.records
    assert record.level == "ERROR"
    assert record.status_message == "ValueError: boom"
    assert record.ended_at is not None


def test_explicit_error_is_not_overwritten() -> None:
    tracer = RecordingTracer()

    def fail() -> None:
        with tracer.observe("step") as obs:
            obs.record_error("specific reason")
            raise RuntimeError("generic")

    with pytest.raises(RuntimeError):
        fail()
    assert tracer.records[0].status_message == "specific reason"


def test_generation_fields_recorded() -> None:
    tracer = RecordingTracer()
    with tracer.observe("llm", kind="generation", model="m") as obs:
        obs.update(usage=Usage(input_tokens=3, output_tokens=4), cost_usd=0.5, metadata={"a": 1})
    (record,) = tracer.records
    assert record.usage == {
        "input_tokens": 3,
        "output_tokens": 4,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
    }
    assert record.cost_usd == 0.5
    assert record.metadata == {"a": 1}


def test_flush_appends_only_new_entries(tmp_path: Path) -> None:
    path = tmp_path / "t" / "trace.jsonl"
    tracer = RecordingTracer(path=path)
    with tracer.observe("one") as obs:
        pass
    assert obs.trace_id is not None
    tracer.score(trace_id=obs.trace_id, name="faithfulness", value=0.5, comment="c")
    tracer.flush()
    with tracer.observe("two"):
        pass
    tracer.flush()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(entry["type"], entry["name"]) for entry in lines] == [
        ("observation", "one"),
        ("score", "faithfulness"),
        ("observation", "two"),
    ]


def test_noop_tracer_claims_no_trace_ids() -> None:
    tracer = NoopTracer()
    with tracer.observe("x") as obs:
        obs.update(output="ignored")
        assert obs.trace_id is None
        assert tracer.current_trace_id() is None
