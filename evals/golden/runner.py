"""Runs a golden set against a target (a model alias plus a system prompt) and scores it.

One model call per case, scored with the deterministic checks in
evals.golden.scoring. A provider error on one case is recorded as that case's
error and the run continues; anything else (budget exhausted, configuration,
bugs) stops the run.

The result is an `EvalRunFile`: the standard run file (manifest, every request
and response or error) plus a reference to the exact golden-set bytes, the
target, per-case check results and a summary. It stands alone as evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from apps.core.cache import CachePolicy
from apps.core.client import ModelClient
from apps.core.errors import ProviderError
from apps.core.evidence import (
    CallRecord,
    RunFile,
    RunManifest,
    budget_record,
    call_settings,
    error_record,
    git_state,
    new_run_id,
    write_run_file,
)
from apps.core.redact import redact
from apps.core.tracing.base import TraceAttributes
from apps.core.types import CompletionRequest, Message, StopReason
from evals.golden.dataset import LoadedGoldenSet
from evals.golden.scoring import CheckResult, score_answer

RUN_KIND = "golden-eval"
DEFAULT_PROMPT_VERSION = "golden-default-v1"
DEFAULT_SYSTEM_PROMPT = (
    "You answer questions about FDA 21 CFR Part 11 (electronic records and electronic "
    "signatures). Answer concisely and cite the section when you know it. If the question "
    "is outside that scope or you are not sure, say that you cannot answer it."
)

CaseStatus = Literal["pass", "fail", "error"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Target(_Model):
    """What is being evaluated. Phase 1a's RAG assistant will become another target."""

    model: str
    system_prompt: str
    prompt_version: str
    max_tokens: int = 512


class GoldenSetRef(_Model):
    name: str
    version: str
    path: str
    sha256: str
    synthetic: bool
    case_count: int


class CaseResult(_Model):
    case_id: str
    tags: tuple[str, ...]
    status: CaseStatus
    call_id: str
    answer: str | None
    stop_reason: StopReason | None
    checks: tuple[CheckResult, ...]
    error: str | None = None


class TagSummary(_Model):
    total: int
    passed: int


class EvalSummary(_Model):
    total: int
    passed: int
    failed: int
    errored: int
    # passed / total: an errored case counts as not passed.
    pass_rate: float
    by_tag: dict[str, TagSummary]


class EvalRunFile(RunFile):
    golden_set: GoldenSetRef
    target: Target
    results: list[CaseResult]
    summary: EvalSummary


def build_requests(loaded: LoadedGoldenSet, target: Target) -> list[CompletionRequest]:
    return [
        CompletionRequest(
            model=target.model,
            system=target.system_prompt,
            messages=(Message.user(case.question),),
            max_tokens=target.max_tokens,
        )
        for case in loaded.golden.cases
    ]


def summarize(results: list[CaseResult]) -> EvalSummary:
    by_tag: dict[str, TagSummary] = {}
    for result in results:
        for tag in result.tags:
            current = by_tag.get(tag, TagSummary(total=0, passed=0))
            by_tag[tag] = TagSummary(
                total=current.total + 1,
                passed=current.passed + (result.status == "pass"),
            )
    total = len(results)
    passed = sum(r.status == "pass" for r in results)
    return EvalSummary(
        total=total,
        passed=passed,
        failed=sum(r.status == "fail" for r in results),
        errored=sum(r.status == "error" for r in results),
        pass_rate=passed / total if total else 0.0,
        by_tag=dict(sorted(by_tag.items())),
    )


def run_golden_eval(
    loaded: LoadedGoldenSet,
    target: Target,
    *,
    client: ModelClient,
    out_dir: Path,
    run_id: str | None = None,
    secrets: list[str] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[EvalRunFile, Path]:
    """Run every case, write <out_dir>/<run_id>/run.json, return (run, path)."""
    golden = loaded.golden
    registry = client.registry
    tracer = client.tracer
    spec = registry.resolve(target.model)
    run_id = run_id or new_run_id(now())
    calls: list[CallRecord] = []
    results: list[CaseResult] = []

    for case, request in zip(golden.cases, build_requests(loaded, target), strict=True):
        call_id = f"case-{case.id}"
        with tracer.observe(
            "golden.case",
            kind="evaluator",
            input={"case_id": case.id, "question": case.question},
            attributes=TraceAttributes(
                prompt_version=target.prompt_version,
                run_id=run_id,
                tags=(RUN_KIND, *case.tags),
            ),
        ) as observation:
            record = CallRecord(
                call_id=call_id,
                trace_id=observation.trace_id,
                settings=call_settings(registry, request),
                request=request,
            )
            try:
                response = client.complete(request)
            except ProviderError as err:
                record = record.model_copy(update={"error": error_record(err)})
                result = CaseResult(
                    case_id=case.id,
                    tags=case.tags,
                    status="error",
                    call_id=call_id,
                    answer=None,
                    stop_reason=None,
                    checks=(),
                    error=redact(f"{type(err).__name__}: {err}"),
                )
            else:
                record = record.model_copy(update={"response": response})
                checks = tuple(
                    score_answer(case, response.text, response.stop_reason, golden.decline_markers)
                )
                result = CaseResult(
                    case_id=case.id,
                    tags=case.tags,
                    status="pass" if all(c.passed for c in checks) else "fail",
                    call_id=call_id,
                    answer=response.text,
                    stop_reason=response.stop_reason,
                    checks=checks,
                )
            observation.update(
                output={
                    "status": result.status,
                    "failed_checks": [c.model_dump() for c in result.checks if not c.passed],
                }
            )
            if observation.trace_id is not None and result.status != "error":
                tracer.score(
                    trace_id=observation.trace_id,
                    name="golden.pass",
                    value=1.0 if result.status == "pass" else 0.0,
                    observation_id=observation.observation_id,
                )
        calls.append(record)
        results.append(result)
    tracer.flush()

    synthetic = golden.synthetic or not spec.is_paid
    notes = []
    if golden.synthetic:
        notes.append("SYNTHETIC golden set: hand-written examples, not owner-labeled data.")
    if not spec.is_paid:
        notes.append("SYNTHETIC answers: scripted fake provider, not model output.")
    if synthetic:
        notes.append("Not evaluation evidence.")

    sha, dirty = git_state()
    run = EvalRunFile(
        manifest=RunManifest(
            run_id=run_id,
            kind=RUN_KIND,
            synthetic=synthetic,
            created_at=now(),
            git_sha=sha,
            git_dirty=dirty,
            models_config_sha256=registry.source_sha256,
            pricing_checked=registry.pricing_checked,
            tracing_backend=tracer.backend,
            cache_policy=CachePolicy.BYPASS,
            prompt_version=target.prompt_version,
            notes=" ".join(notes),
        ),
        calls=calls,
        budget=budget_record(client.guard.snapshot() if client.guard else None),
        golden_set=GoldenSetRef(
            name=golden.name,
            version=golden.version,
            path=loaded.path,
            sha256=loaded.sha256,
            synthetic=golden.synthetic,
            case_count=len(golden.cases),
        ),
        target=target,
        results=results,
        summary=summarize(results),
    )
    path = out_dir / run_id / "run.json"
    write_run_file(path, run, secrets)
    return run, path


def read_eval_run_file(path: Path) -> EvalRunFile:
    return EvalRunFile.model_validate_json(path.read_text(encoding="utf-8"))
