"""Command-line plumbing shared by every command that may call a paid API.

    --allow-paid             required for any paid call; keys alone enable nothing
    --unattended             never prompt; use limits from --budget-profile
    --budget-profile NAME    profile in config/budgets.yaml (unattended runs)

A run is interactive only when nobody asked for unattended mode, no CI marker is
set and stdin is a terminal. Everything else is unattended: it never prompts and
must stay within an approved profile from config/budgets.yaml. Omitting
--unattended in CI therefore cannot fall back to the interactive path.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from apps.core.budget import (
    BudgetGuard,
    BudgetsConfig,
    PlannedCalls,
    RunMode,
    authorize_run,
    estimate_plan,
)
from apps.core.models import ModelRegistry

# Environment variables that CI systems set. Any non-empty value other than
# "false"/"0" marks the run as unattended.
CI_MARKERS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
    "TF_BUILD",
)


@dataclass(frozen=True)
class ModeDecision:
    mode: RunMode
    reason: str


def add_paid_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("paid execution")
    group.add_argument(
        "--allow-paid",
        action="store_true",
        help="allow calls to paid APIs (otherwise only the fake provider runs)",
    )
    group.add_argument(
        "--unattended",
        action="store_true",
        help="never prompt; stay within the limits of --budget-profile (implied in CI)",
    )
    group.add_argument("--budget-profile", default=None, help="profile in config/budgets.yaml")


def detect_run_mode(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
    stdin_is_tty: bool | None = None,
) -> ModeDecision:
    env = os.environ if environ is None else environ
    if args.unattended:
        return ModeDecision("unattended", "--unattended was given")
    for name in CI_MARKERS:
        value = env.get(name, "").strip().lower()
        if value and value not in {"false", "0"}:
            return ModeDecision("unattended", f"CI environment detected: {name} is set")
    tty = sys.stdin.isatty() if stdin_is_tty is None else stdin_is_tty
    if not tty:
        return ModeDecision("unattended", "stdin is not a terminal")
    return ModeDecision("interactive", "terminal session")


def confirm_on_terminal(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    print(question, file=sys.stderr)
    return input("Type 'yes' to proceed: ").strip().lower() == "yes"


def prepare_paid_run(
    args: argparse.Namespace,
    plan: list[PlannedCalls],
    *,
    registry: ModelRegistry,
    budgets: BudgetsConfig,
    confirm: Callable[[str], bool] = confirm_on_terminal,
    environ: Mapping[str, str] | None = None,
    stdin_is_tty: bool | None = None,
) -> BudgetGuard | None:
    """Return a guard for an authorized paid run, or None when paid calls are off."""
    if not args.allow_paid:
        return None
    decision = detect_run_mode(args, environ=environ, stdin_is_tty=stdin_is_tty)
    estimate = estimate_plan(plan, registry)
    limits = authorize_run(
        estimate,
        mode=decision.mode,
        budgets=budgets,
        profile=args.budget_profile,
        confirm=confirm,
        mode_reason=decision.reason,
    )
    return BudgetGuard(limits)
