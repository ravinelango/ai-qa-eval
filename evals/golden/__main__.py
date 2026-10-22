"""Golden-set evaluator CLI.

    make golden                                   (synthetic sample, fake provider, offline)
    uv run python -m evals.golden --golden PATH --model ALIAS [--allow-paid ...]

With a fake model alias (the default) answers come from a scripted replies file
(--fake-replies) and the run is labeled synthetic. A paid alias needs
--allow-paid and approved budget limits, like every paid command
(docs/decisions/0004-paid-execution-and-budgets.md).

Exit status: 0 when the run completes (and meets --min-pass-rate if given),
1 when the pass rate is below --min-pass-rate, 2 on usage or configuration errors.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from apps.core.budget import BudgetsConfig, PlannedCalls
from apps.core.client import ModelClient
from apps.core.errors import ConfigurationError, QAEvalError
from apps.core.evidence import new_run_id
from apps.core.factory import build_model_client, build_tracer
from apps.core.models import REPO_ROOT, ModelRegistry
from apps.core.paid_cli import add_paid_arguments, prepare_paid_run
from apps.core.pricing import input_tokens_upper_bound
from apps.core.providers.fake import FakeBackend, ScriptedReply
from apps.core.redact import configure_logging
from apps.core.settings import Settings
from apps.core.types import StopReason
from evals.golden.dataset import LoadedGoldenSet, load_golden_set
from evals.golden.runner import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SYSTEM_PROMPT,
    EvalRunFile,
    Target,
    build_requests,
    run_golden_eval,
)

DATASETS = Path(__file__).parent / "datasets"
SAMPLE_GOLDEN = DATASETS / "sample.yaml"
SAMPLE_FAKE_REPLIES = DATASETS / "sample.fake-replies.yaml"


class FakeReply(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    stop_reason: StopReason = "end_turn"


def load_fake_replies(path: Path, loaded: LoadedGoldenSet) -> list[ScriptedReply]:
    """One scripted reply per case, in case order. Every case must have one."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        replies = {k: FakeReply.model_validate(v) for k, v in dict(raw).items()}
    except (OSError, yaml.YAMLError, ValidationError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid fake replies file {path}: {exc}") from exc
    ids = [case.id for case in loaded.golden.cases]
    missing = [i for i in ids if i not in replies]
    unknown = sorted(set(replies) - set(ids))
    if missing or unknown:
        raise ConfigurationError(
            f"fake replies {path} do not match the golden set: missing {missing}, unknown {unknown}"
        )
    return [ScriptedReply.text(replies[i].text, stop_reason=replies[i].stop_reason) for i in ids]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--golden", type=Path, default=SAMPLE_GOLDEN, help="golden set YAML")
    parser.add_argument("--model", default="fake", help="model alias from config/models.yaml")
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="system prompt for the target (default: the built-in Part 11 prompt)",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="label recorded in the run (default: derived from the prompt)",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--fake-replies",
        type=Path,
        default=None,
        help="scripted replies for a fake model (default: the sample's replies)",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "golden")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help="exit 1 if the pass rate (0-1) is below this",
    )
    add_paid_arguments(parser)
    return parser


def resolve_target(args: argparse.Namespace) -> Target:
    if args.system_prompt_file is None:
        prompt, default_version = DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPT_VERSION
    else:
        prompt = args.system_prompt_file.read_text(encoding="utf-8")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        default_version = f"file-sha256-{digest}"
    return Target(
        model=args.model,
        system_prompt=prompt,
        prompt_version=args.prompt_version or default_version,
        max_tokens=args.max_tokens,
    )


def build_client(
    args: argparse.Namespace,
    loaded: LoadedGoldenSet,
    target: Target,
    *,
    settings: Settings,
    registry: ModelRegistry,
    run_id: str,
) -> ModelClient:
    tracer = build_tracer(settings, run_id=run_id)
    spec = registry.resolve(target.model)
    if not spec.is_paid:
        replies_path = args.fake_replies or SAMPLE_FAKE_REPLIES
        backend = FakeBackend(load_fake_replies(replies_path, loaded), name=spec.provider)
        return ModelClient(registry=registry, backends={spec.provider: backend}, tracer=tracer)
    if not args.allow_paid:
        raise ConfigurationError(
            f"model alias {spec.alias!r} calls a paid API; pass --allow-paid to run it"
        )
    requests = build_requests(loaded, target)
    plan = [
        PlannedCalls(
            purpose="golden-set generation",
            model_alias=target.model,
            calls=len(requests),
            max_input_tokens=max(input_tokens_upper_bound(r) for r in requests),
            max_output_tokens=target.max_tokens,
        )
    ]
    guard = prepare_paid_run(args, plan, registry=registry, budgets=BudgetsConfig.load())
    return build_model_client(
        settings, registry=registry, tracer=tracer, allow_paid=True, guard=guard
    )


def render(run: EvalRunFile, path: Path) -> str:
    lines = []
    if run.manifest.synthetic:
        lines.append(f"SYNTHETIC RUN - {run.manifest.notes}")
    g, t = run.golden_set, run.target
    lines.append(
        f"golden set: {g.name} v{g.version} ({g.case_count} cases, sha256 {g.sha256[:12]})"
    )
    lines.append(f"target:     model={t.model} prompt_version={t.prompt_version}")
    for result in run.results:
        failed = [f"{c.kind}: {c.expected}" for c in result.checks if not c.passed]
        detail = result.error or "; ".join(failed)
        lines.append(f"  {result.status.upper():5} {result.case_id}  {detail}".rstrip())
    s = run.summary
    lines.append(
        f"passed {s.passed}/{s.total} ({s.pass_rate:.0%}); failed {s.failed}; errored {s.errored}"
    )
    for tag, ts in s.by_tag.items():
        lines.append(f"  tag {tag}: {ts.passed}/{ts.total}")
    lines.append(f"run file: {path}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    secrets = settings.secret_values()
    configure_logging(secrets)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_pass_rate is not None and not 0.0 <= args.min_pass_rate <= 1.0:
        parser.error("--min-pass-rate must be between 0 and 1")
    try:
        registry = ModelRegistry.load()
        loaded = load_golden_set(args.golden)
        target = resolve_target(args)
        run_id = new_run_id()
        client = build_client(
            args, loaded, target, settings=settings, registry=registry, run_id=run_id
        )
        run, path = run_golden_eval(
            loaded, target, client=client, out_dir=args.out, run_id=run_id, secrets=secrets
        )
    except (QAEvalError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(render(run, path))
    if args.min_pass_rate is not None and run.summary.pass_rate < args.min_pass_rate:
        print(
            f"pass rate {run.summary.pass_rate:.0%} is below --min-pass-rate "
            f"{args.min_pass_rate:.0%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
