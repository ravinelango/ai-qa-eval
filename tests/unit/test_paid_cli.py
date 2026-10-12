from __future__ import annotations

import argparse

import pytest

from apps.core.budget import BudgetsConfig, PlannedCalls
from apps.core.errors import BudgetConfigError, PlanRejectedError
from apps.core.models import ModelRegistry
from apps.core.paid_cli import add_paid_arguments, prepare_paid_run

PLAN = [
    PlannedCalls(
        purpose="generation",
        model_alias="opus",
        calls=30,
        max_input_tokens=1_000,
        max_output_tokens=500,
    )
]


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_paid_arguments(parser)
    return parser.parse_args(list(argv))


def test_paid_execution_is_off_by_default(registry: ModelRegistry) -> None:
    guard = prepare_paid_run(_args(), PLAN, registry=registry, budgets=BudgetsConfig.load())
    assert guard is None


def test_interactive_paid_run_over_threshold_requires_confirmation(registry: ModelRegistry) -> None:
    with pytest.raises(PlanRejectedError):
        prepare_paid_run(
            _args("--allow-paid"),
            PLAN,
            registry=registry,
            budgets=BudgetsConfig.load(),
            confirm=lambda _: False,
        )
    guard = prepare_paid_run(
        _args("--allow-paid"),
        PLAN,
        registry=registry,
        budgets=BudgetsConfig.load(),
        confirm=lambda _: True,
    )
    assert guard is not None
    assert guard.limits.max_request_attempts == 90  # 30 calls x (1 + 2 retries)


def test_unattended_run_with_shipped_profile_is_blocked(registry: ModelRegistry) -> None:
    with pytest.raises(BudgetConfigError, match="no approved limits"):
        prepare_paid_run(
            _args("--allow-paid", "--unattended", "--budget-profile", "ci-eval-smoke"),
            PLAN,
            registry=registry,
            budgets=BudgetsConfig.load(),
        )
