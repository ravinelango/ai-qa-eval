"""Saved run files: the primary evidence for every run.

A run file must be readable on its own, without Langfuse access: it stores the
full requests, the responses or errors, measurements, configuration/version
identifiers and the trace ID (a supplementary link only).

`synthetic=True` marks runs produced by the fake provider. Synthetic runs
demonstrate plumbing; they are never evaluation evidence. Phase 1 extends this
format with per-case scores.
"""

from __future__ import annotations

import secrets
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from apps.core.budget import BudgetSnapshot
from apps.core.cache import CachePolicy
from apps.core.errors import ProviderError
from apps.core.models import REPO_ROOT, ModelRegistry
from apps.core.redact import redact
from apps.core.types import CompletionRequest, CompletionResponse, FailedAttempt

# 2: CallRecord.settings, failed-attempt history, outstanding reservations.
RUN_FILE_SCHEMA_VERSION = 2


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunManifest(_Model):
    run_id: str
    kind: str
    synthetic: bool
    created_at: datetime
    git_sha: str | None
    git_dirty: bool | None
    models_config_sha256: str
    pricing_checked: str
    tracing_backend: str
    cache_policy: CachePolicy
    prompt_version: str | None = None
    notes: str = ""


class CallSettings(_Model):
    """The resolved values a call ran with. The config hash identifies them; this
    preserves them, so the run file can be read without the config at that commit."""

    provider: str
    model_alias: str
    model_id: str
    effort: str | None
    max_tokens: int
    temperature: float | None
    timeout_s: float
    max_retries: int
    backoff_initial_s: float
    backoff_max_s: float
    pricing_per_mtok: dict[str, Any] | None
    pricing_checked: str


class CallRecord(_Model):
    call_id: str
    trace_id: str | None
    settings: CallSettings
    request: CompletionRequest
    # On success the attempt history is in response.execution.failed_attempts;
    # on failure it is in error["failed_attempts"].
    response: CompletionResponse | None = None
    error: dict[str, Any] | None = None


def call_settings(registry: ModelRegistry, request: CompletionRequest) -> CallSettings:
    spec = registry.resolve(request.model)
    provider = registry.provider_settings(spec.provider)
    return CallSettings(
        provider=spec.provider,
        model_alias=spec.alias,
        model_id=spec.model_id,
        effort=spec.effort,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        timeout_s=provider.timeout_s,
        max_retries=provider.max_retries,
        backoff_initial_s=provider.backoff_initial_s,
        backoff_max_s=provider.backoff_max_s,
        pricing_per_mtok=spec.pricing.model_dump() if spec.pricing else None,
        pricing_checked=registry.pricing_checked,
    )


class RunFile(_Model):
    schema_version: int = RUN_FILE_SCHEMA_VERSION
    manifest: RunManifest
    calls: list[CallRecord]
    budget: dict[str, Any] | None = None


def new_run_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def git_state(repo: Path = REPO_ROOT) -> tuple[str | None, bool | None]:
    """(commit SHA, has uncommitted changes); (None, None) outside a git checkout."""
    git = shutil.which("git")
    if git is None:
        return None, None
    try:
        sha = subprocess.run(  # noqa: S603 - fixed argv, no external input
            [git, "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603 - fixed argv, no external input
            [git, "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return sha, bool(status.strip())


def error_record(error: BaseException) -> dict[str, Any]:
    record: dict[str, Any] = (
        error.describe()
        if isinstance(error, ProviderError)
        else {"error_type": type(error).__name__, "message": str(error)}
    )
    failed: tuple[FailedAttempt, ...] = getattr(error, "failed_attempts", ())
    record["failed_attempts"] = [a.model_dump() for a in failed]
    return record


def budget_record(snapshot: BudgetSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "attempts_dispatched": snapshot.attempts_dispatched,
        "settled_cost_usd": snapshot.settled_cost_usd,
        "uncertain_cost_usd": snapshot.uncertain_cost_usd,
        # Non-zero means an attempt never reached a settled state.
        "outstanding_reservations_usd": snapshot.outstanding_reservations_usd,
        "committed_cost_usd": snapshot.committed_cost_usd,
        "max_request_attempts": snapshot.limits.max_request_attempts,
        "max_estimated_cost_usd": snapshot.limits.max_estimated_cost_usd,
        "note": "estimates from the dated pricing table, not the provider's bill",
    }


def write_run_file(path: Path, run: RunFile, secrets: list[str] | None = None) -> None:
    """Write the run file, masking configured secrets and credential-shaped strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact(run.model_dump_json(indent=2), secrets or ()) + "\n", encoding="utf-8")


def read_run_file(path: Path) -> RunFile:
    return RunFile.model_validate_json(path.read_text(encoding="utf-8"))
