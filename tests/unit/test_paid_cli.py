from __future__ import annotations

import argparse

import pytest

from apps.core.budget import BudgetsConfig, PlannedCalls, ProfileLimits
from apps.core.errors import BudgetConfigError, PlanRejectedError
from apps.core.models import ModelRegistry
from apps.core.paid_cli import CI_MARKERS, add_paid_arguments, detect_run_mode, prepare_paid_run

PLAN = [
    PlannedCalls(
        purpose="generation",
        model_alias="opus",
        calls=30,
        max_input_tokens=1_000,
        max_output_tokens=500,
    )
]
ONE_HAIKU_CALL = [
    PlannedCalls(
        purpose="generation",
        model_alias="haiku",
        calls=1,
        max_input_tokens=1_000,
        max_output_tokens=100,
    )
]
TERMINAL: dict[str, str] = {}  # an environment with no CI markers


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_paid_arguments(parser)
    return parser.parse_args(list(argv))


def _never(_: str) -> bool:
    raise AssertionError("an unattended run must not prompt")


def test_paid_execution_is_off_by_default(registry: ModelRegistry) -> None:
    guard = prepare_paid_run(_args(), PLAN, registry=registry, budgets=BudgetsConfig.load())
    assert guard is None


# --- mode detection ------------------------------------------------------------------


def test_terminal_without_ci_markers_is_interactive() -> None:
    decision = detect_run_mode(_args(), environ=TERMINAL, stdin_is_tty=True)
    assert decision.mode == "interactive"


def test_explicit_flag_is_unattended() -> None:
    decision = detect_run_mode(_args("--unattended"), environ=TERMINAL, stdin_is_tty=True)
    assert decision.mode == "unattended"


@pytest.mark.parametrize("marker", CI_MARKERS)
def test_any_ci_marker_makes_the_run_unattended(marker: str) -> None:
    decision = detect_run_mode(_args(), environ={marker: "true"}, stdin_is_tty=True)
    assert decision.mode == "unattended"
    assert marker in decision.reason


@pytest.mark.parametrize("value", ["false", "0", ""])
def test_falsey_ci_marker_alone_does_not_force_unattended(value: str) -> None:
    decision = detect_run_mode(_args(), environ={"CI": value}, stdin_is_tty=True)
    assert decision.mode == "interactive"


def test_non_terminal_stdin_is_unattended() -> None:
    decision = detect_run_mode(_args(), environ=TERMINAL, stdin_is_tty=False)
    assert decision.mode == "unattended"
    assert "terminal" in decision.reason


# --- regression: review finding 1 ---------------------------------------------------


def test_ci_without_unattended_flag_cannot_use_interactive_path(registry: ModelRegistry) -> None:
    """CI=true, --allow-paid, no --unattended, one Haiku call: the shipped (null)
    profiles must block it. Before the fix this produced a guard allowing 3 attempts."""
    with pytest.raises(BudgetConfigError, match=r"CI environment detected \(CI is set\)"):
        prepare_paid_run(
            _args("--allow-paid"),
            ONE_HAIKU_CALL,
            registry=registry,
            budgets=BudgetsConfig.load(),
            confirm=_never,
            environ={"CI": "true"},
            stdin_is_tty=True,
        )


@pytest.mark.parametrize("profile", ["ci-eval-smoke", "ci-eval-full"])
def test_shipped_ci_profiles_block_detected_ci_runs(registry: ModelRegistry, profile: str) -> None:
    with pytest.raises(BudgetConfigError, match="no approved limits"):
        prepare_paid_run(
            _args("--allow-paid", "--budget-profile", profile),
            ONE_HAIKU_CALL,
            registry=registry,
            budgets=BudgetsConfig.load(),
            confirm=_never,
            environ={"GITHUB_ACTIONS": "true"},
            stdin_is_tty=False,
        )


def test_detected_ci_run_uses_approved_profile_without_prompting(registry: ModelRegistry) -> None:
    budgets = BudgetsConfig(
        approval_threshold_attempts=50,
        profiles={"approved": ProfileLimits(max_request_attempts=5, max_estimated_cost_usd=1.0)},
    )
    guard = prepare_paid_run(
        _args("--allow-paid", "--budget-profile", "approved"),
        ONE_HAIKU_CALL,
        registry=registry,
        budgets=budgets,
        confirm=_never,
        environ={"CI": "1"},
        stdin_is_tty=False,
    )
    assert guard is not None
    assert guard.limits.max_request_attempts == 5


# --- interactive ----------------------------------------------------------------------


def test_interactive_paid_run_over_threshold_requires_confirmation(registry: ModelRegistry) -> None:
    budgets = BudgetsConfig.load()
    with pytest.raises(PlanRejectedError):
        prepare_paid_run(
            _args("--allow-paid"),
            PLAN,
            registry=registry,
            budgets=budgets,
            confirm=lambda _: False,
            environ=TERMINAL,
            stdin_is_tty=True,
        )
    guard = prepare_paid_run(
        _args("--allow-paid"),
        PLAN,
        registry=registry,
        budgets=budgets,
        confirm=lambda _: True,
        environ=TERMINAL,
        stdin_is_tty=True,
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
            confirm=_never,
        )
