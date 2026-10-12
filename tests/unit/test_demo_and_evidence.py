"""The offline demo runs without keys or network and writes self-contained evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from apps.core.demo import BANNER, run_demo
from apps.core.errors import ProviderTimeoutError
from apps.core.evidence import error_record, read_run_file

EMBEDDING_MODULES = ("sentence_transformers", "torch", "chromadb", "fastembed", "onnxruntime")


def test_demo_runs_offline_and_writes_synthetic_run_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    run = run_demo(tmp_path)
    run_dir = tmp_path / run.manifest.run_id

    saved = read_run_file(run_dir / "run.json")
    assert saved == run
    assert saved.manifest.synthetic is True
    assert saved.manifest.notes == BANNER
    assert saved.manifest.cache_policy == "bypass"
    assert saved.manifest.models_config_sha256
    assert [c.call_id for c in saved.calls] == ["turn-1", "turn-2"]

    first, second = (c.response for c in saved.calls)
    assert first is not None
    assert second is not None
    assert first.stop_reason == "tool_use"
    assert first.execution.attempts == 2  # scripted 429, then success
    assert second.text.startswith("[SCRIPTED FAKE REPLY]")
    # The second request carries the tool call and its result back to the model.
    roles = [m.role for m in saved.calls[1].request.messages]
    assert roles == ["user", "assistant", "user"]


def test_demo_trace_file_matches_run_file(tmp_path: Path) -> None:
    run = run_demo(tmp_path)
    lines = (tmp_path / run.manifest.run_id / "trace.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    trace_ids = {r["trace_id"] for r in records}
    assert trace_ids == {run.calls[0].trace_id}
    generations = [r for r in records if r["kind"] == "generation"]
    assert len(generations) == 2
    assert generations[0]["metadata"]["attempt_errors"][0]["error_type"] == "ProviderRateLimitError"


def test_demo_does_not_load_embedding_models(tmp_path: Path) -> None:
    run_demo(tmp_path)
    assert not [m for m in EMBEDDING_MODULES if m in sys.modules]


def test_error_record_keeps_charge_status_and_no_usage() -> None:
    record = error_record(ProviderTimeoutError("timed out", provider="anthropic"))
    assert record["charge"] == "uncertain"
    assert record["retryable"] is True
    assert "usage" not in record
