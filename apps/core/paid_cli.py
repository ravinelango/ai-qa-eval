"""Command-line plumbing shared by every command that may call a paid API.

    --allow-paid             required for any paid call; keys alone enable nothing
    --unattended             never prompt; use limits from --budget-profile
    --budget-profile NAME    profile in config/budgets.yaml (unattended runs)

Interactive runs over the approval threshold print the estimate and wait for "yes".
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from apps.core.budget import (
    BudgetGuard,
    BudgetsConfig,
    PlannedCalls,
    authorize_run,
    estimate_plan,
)
from apps.core.models import ModelRegistry


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
        help="never prompt; stay within the limits of --budget-profile (for CI)",
    )
    group.add_argument("--budget-profile", default=None, help="profile in config/budgets.yaml")


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
) -> BudgetGuard | None:
    """Return a guard for an authorized paid run, or None when paid calls are off."""
    if not args.allow_paid:
        return None
    estimate = estimate_plan(plan, registry)
    limits = authorize_run(
        estimate,
        mode="unattended" if args.unattended else "interactive",
        budgets=budgets,
        profile=args.budget_profile,
        confirm=confirm,
    )
    return BudgetGuard(limits)
