"""The offline demo runs without keys or network and writes self-contained evidence."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from apps.core.budget import BudgetGuard, BudgetLimits
from apps.core.cache import CachePolicy
from apps.core.client import ModelClient
from apps.core.demo import BANNER, run_demo
from apps.core.errors import ProviderBadRequestError, ProviderTimeoutError
from apps.core.evidence import (
    CallRecord,
    RunFile,
    RunManifest,
    budget_record,
    call_settings,
    error_record,
    read_run_file,
    write_run_file,
)
from apps.core.models import ModelRegistry
from apps.core.providers.fake import FakeBackend, Step
from apps.core.tracing.local import NoopTracer
from tests.helpers import simple_request

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
    assert (
        generations[0]["metadata"]["failed_attempts"][0]["error_type"] == "ProviderRateLimitError"
    )


def test_demo_does_not_load_embedding_models(tmp_path: Path) -> None:
    run_demo(tmp_path)
    assert not [m for m in EMBEDDING_MODULES if m in sys.modules]


def test_error_record_keeps_charge_status_and_no_usage() -> None:
    record = error_record(ProviderTimeoutError("timed out", provider="anthropic"))
    assert record["charge"] == "uncertain"
    assert record["retryable"] is True
    assert "usage" not in record


# --- regression: review finding 4 (run file stands alone) -----------------------------


def test_run_file_alone_explains_the_retry_with_noop_tracer(tmp_path: Path) -> None:
    run = run_demo(tmp_path, tracer=NoopTracer())
    run_dir = tmp_path / run.manifest.run_id
    assert not (run_dir / "trace.jsonl").exists()

    saved = read_run_file(run_dir / "run.json")
    assert saved.manifest.tracing_backend == "none"
    first = saved.calls[0]
    assert first.trace_id is None
    assert first.response is not None
    (failed,) = first.response.execution.failed_attempts
    assert failed.error_type == "ProviderRateLimitError"
    assert failed.status_code == 429
    assert failed.message == "HTTP 429 (scripted)"
    assert first.response.execution.attempts == 2


def test_run_file_preserves_resolved_settings(tmp_path: Path, registry: ModelRegistry) -> None:
    run = run_demo(tmp_path, tracer=NoopTracer())
    settings = read_run_file(tmp_path / run.manifest.run_id / "run.json").calls[0].settings
    fake = registry.resolve("fake")
    provider = registry.provider_settings("fake")
    assert settings.model_id == fake.model_id
    assert settings.effort == fake.effort
    assert settings.max_retries == provider.max_retries
    assert settings.timeout_s == provider.timeout_s
    assert settings.max_tokens == 256
    assert settings.pricing_checked == registry.pricing_checked


def test_call_settings_resolve_paid_model_values(registry: ModelRegistry) -> None:
    settings = call_settings(registry, simple_request("opus", max_tokens=321))
    assert settings.provider == "anthropic"
    assert settings.model_id == "claude-opus-5-5"
    assert settings.effort == "medium"
    assert settings.max_tokens == 321
    assert settings.pricing_per_mtok is not None
    assert settings.pricing_per_mtok["input"] == 4.0


def test_failed_call_record_keeps_every_attempt(tmp_path: Path, registry: ModelRegistry) -> None:
    script: list[Step] = [
        ProviderTimeoutError("timed out", provider="fake"),
        ProviderBadRequestError("HTTP 400", provider="fake", status_code=400),
    ]
    client = ModelClient(
        registry=registry, backends={"fake": FakeBackend(script)}, sleep=lambda _: None
    )
    request = simple_request("fake")
    with pytest.raises(ProviderBadRequestError) as info:
        client.complete(request)
    record = CallRecord(
        call_id="c1",
        trace_id=None,
        settings=call_settings(registry, request),
        request=request,
        error=error_record(info.value),
    )
    path = tmp_path / "run.json"
    write_run_file(path, _run_file([record]))
    saved = read_run_file(path).calls[0].error
    assert saved is not None
    assert saved["error_type"] == "ProviderBadRequestError"
    assert [a["error_type"] for a in saved["failed_attempts"]] == [
        "ProviderTimeoutError",
        "ProviderBadRequestError",
    ]
    assert saved["failed_attempts"][0]["charge"] == "uncertain"


def test_budget_record_does_not_hide_outstanding_reservations(registry: ModelRegistry) -> None:
    guard = BudgetGuard(BudgetLimits(5, 1.0))
    guard.reserve(registry.resolve("opus"), simple_request("opus"))  # never settled
    record = budget_record(guard.snapshot())
    assert record is not None
    assert record["outstanding_reservations_usd"] > 0
    assert record["committed_cost_usd"] == pytest.approx(record["outstanding_reservations_usd"])


def test_run_file_masks_credential_shaped_strings(tmp_path: Path, registry: ModelRegistry) -> None:
    leaked = "sk-ant-api03-SYNTHETICSYNTHETIC"  # gitleaks:allow - synthetic
    request = simple_request("fake", text=f"please ignore {leaked}")
    record = CallRecord(
        call_id="c1", trace_id=None, settings=call_settings(registry, request), request=request
    )
    path = tmp_path / "run.json"
    write_run_file(path, _run_file([record]))
    assert leaked not in path.read_text()


def _run_file(calls: list[CallRecord]) -> RunFile:
    return RunFile(
        manifest=RunManifest(
            run_id="test",
            kind="test",
            synthetic=True,
            created_at=datetime(2026, 10, 12, tzinfo=UTC),
            git_sha=None,
            git_dirty=None,
            models_config_sha256="x",
            pricing_checked="2026-10-12",
            tracing_backend="none",
            cache_policy=CachePolicy.BYPASS,
        ),
        calls=calls,
    )
