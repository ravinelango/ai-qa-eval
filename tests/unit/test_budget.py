from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from apps.core.budget import (
    BudgetGuard,
    BudgetLimits,
    BudgetsConfig,
    PlannedCalls,
    ProfileLimits,
    authorize_run,
    estimate_plan,
)
from apps.core.errors import (
    BudgetConfigError,
    BudgetExceededError,
    PlanRejectedError,
    ProviderBadRequestError,
    ProviderTimeoutError,
)
from apps.core.models import ModelRegistry
from apps.core.pricing import input_tokens_upper_bound, worst_case_attempt_cost
from apps.core.types import Usage
from tests.helpers import simple_request

BUDGETS = BudgetsConfig(
    approval_threshold_attempts=50,
    profiles={
        "ci": ProfileLimits(max_request_attempts=100, max_estimated_cost_usd=5.0),
        "unset": ProfileLimits(),
    },
)


def _plan(calls: int, alias: str = "opus") -> list[PlannedCalls]:
    return [
        PlannedCalls(
            purpose="generation",
            model_alias=alias,
            calls=calls,
            max_input_tokens=1_000,
            max_output_tokens=500,
        )
    ]


# --- estimates -----------------------------------------------------------------------


def test_estimate_counts_every_line_and_retries(registry: ModelRegistry) -> None:
    plan = [
        PlannedCalls(
            purpose="generation",
            model_alias="opus",
            calls=20,
            max_input_tokens=1_000,
            max_output_tokens=500,
        ),
        # judging 20 answers on 2 metrics, repeated 3 times for variance
        PlannedCalls(
            purpose="judge",
            model_alias="opus",
            calls=20 * 2 * 3,
            max_input_tokens=2_000,
            max_output_tokens=300,
        ),
    ]
    est = estimate_plan(plan, registry)
    retries = registry.provider_settings("anthropic").max_retries
    assert est.attempts_no_retry == 140
    assert est.attempts_all_retries == 140 * (1 + retries)
    pricing = registry.resolve("opus").pricing
    assert pricing is not None
    per_gen = worst_case_attempt_cost(pricing, 1_000, 500)
    per_judge = worst_case_attempt_cost(pricing, 2_000, 300)
    assert est.cost_no_retry_usd == pytest.approx(20 * per_gen + 120 * per_judge)
    assert "TOTAL" in est.render()


def test_estimate_blocks_paid_model_without_pricing(
    models_yaml: dict[str, Any], write_models: Callable[[dict[str, Any]], Path]
) -> None:
    data = copy.deepcopy(models_yaml)
    del data["models"]["opus"]["pricing"]
    registry = ModelRegistry.load(write_models(data))
    with pytest.raises(BudgetConfigError, match="no pricing"):
        estimate_plan(_plan(1), registry)


def test_fake_model_costs_nothing(registry: ModelRegistry) -> None:
    assert estimate_plan(_plan(10, alias="fake"), registry).cost_all_retries_usd == 0.0


# --- authorization -------------------------------------------------------------------


def test_interactive_small_plan_needs_no_prompt(registry: ModelRegistry) -> None:
    est = estimate_plan(_plan(5), registry)

    def never(_: str) -> bool:
        raise AssertionError("should not prompt")

    limits = authorize_run(est, mode="interactive", budgets=BUDGETS, confirm=never)
    assert limits.max_request_attempts == est.attempts_all_retries
    assert limits.max_estimated_cost_usd == pytest.approx(est.cost_all_retries_usd)


def test_interactive_large_plan_shows_estimate_and_needs_yes(registry: ModelRegistry) -> None:
    est = estimate_plan(_plan(40), registry)  # 40 calls x 3 attempts > 50
    seen: list[str] = []

    def approve(question: str) -> bool:
        seen.append(question)
        return True

    authorize_run(est, mode="interactive", budgets=BUDGETS, confirm=approve)
    assert len(seen) == 1
    assert str(est.attempts_all_retries) in seen[0]
    assert "not the provider's bill" in seen[0]


def test_interactive_decline_rejects_plan(registry: ModelRegistry) -> None:
    est = estimate_plan(_plan(40), registry)
    with pytest.raises(PlanRejectedError, match="declined"):
        authorize_run(est, mode="interactive", budgets=BUDGETS, confirm=lambda _: False)


def test_interactive_without_prompt_capability_rejects(registry: ModelRegistry) -> None:
    est = estimate_plan(_plan(40), registry)
    with pytest.raises(PlanRejectedError):
        authorize_run(est, mode="interactive", budgets=BUDGETS, confirm=None)


def test_unattended_never_prompts_and_uses_profile(registry: ModelRegistry) -> None:
    est = estimate_plan(_plan(40), registry)

    def never(_: str) -> bool:
        raise AssertionError("unattended runs must not prompt")

    limits = authorize_run(est, mode="unattended", budgets=BUDGETS, profile="ci", confirm=never)
    assert limits == BudgetLimits(100, 5.0)


@pytest.mark.parametrize("profile", [None, "missing", "unset"])
def test_unattended_without_approved_limits_stops(
    registry: ModelRegistry, profile: str | None
) -> None:
    est = estimate_plan(_plan(1), registry)
    with pytest.raises(BudgetConfigError):
        authorize_run(est, mode="unattended", budgets=BUDGETS, profile=profile)


def test_unattended_rejects_plan_over_attempt_limit(registry: ModelRegistry) -> None:
    est = estimate_plan(_plan(101), registry)
    with pytest.raises(PlanRejectedError, match="attempts"):
        authorize_run(est, mode="unattended", budgets=BUDGETS, profile="ci")


def test_unattended_rejects_plan_over_cost_limit(registry: ModelRegistry) -> None:
    plan = [
        PlannedCalls(
            purpose="generation",
            model_alias="opus",
            calls=50,
            max_input_tokens=100_000,
            max_output_tokens=16_000,
        )
    ]
    est = estimate_plan(plan, registry)
    assert est.cost_no_retry_usd > 5.0
    with pytest.raises(PlanRejectedError, match=r"exceeds"):
        authorize_run(est, mode="unattended", budgets=BUDGETS, profile="ci")


def test_shipped_budget_profiles_block_paid_runs_until_owner_sets_limits(
    registry: ModelRegistry,
) -> None:
    shipped = BudgetsConfig.load()
    est = estimate_plan(_plan(1), registry)
    for name in shipped.profiles:
        with pytest.raises(BudgetConfigError):
            authorize_run(est, mode="unattended", budgets=shipped, profile=name)


# --- per-attempt enforcement ----------------------------------------------------------


def _reservation_cost(registry: ModelRegistry, max_tokens: int = 100) -> float:
    pricing = registry.resolve("opus").pricing
    assert pricing is not None
    request = simple_request("opus", max_tokens=max_tokens)
    return worst_case_attempt_cost(pricing, input_tokens_upper_bound(request), max_tokens)


def test_guard_stops_before_exceeding_attempt_limit(registry: ModelRegistry) -> None:
    guard = BudgetGuard(BudgetLimits(max_request_attempts=2, max_estimated_cost_usd=100.0))
    spec, request = registry.resolve("opus"), simple_request("opus")
    usage = Usage(input_tokens=10, output_tokens=10)
    guard.reserve(spec, request).settle_success(usage)
    guard.reserve(spec, request).settle_success(usage)
    with pytest.raises(BudgetExceededError, match="attempt limit"):
        guard.reserve(spec, request)
    assert guard.snapshot().attempts_dispatched == 2


def test_guard_stops_before_exceeding_cost_limit(registry: ModelRegistry) -> None:
    per_attempt = _reservation_cost(registry)
    guard = BudgetGuard(
        BudgetLimits(max_request_attempts=100, max_estimated_cost_usd=per_attempt * 1.5)
    )
    spec, request = registry.resolve("opus"), simple_request("opus")
    first = guard.reserve(spec, request)
    # The first reservation is still outstanding, so a second worst case does not fit.
    with pytest.raises(BudgetExceededError, match="could cost"):
        guard.reserve(spec, request)
    first.settle_success(Usage(input_tokens=1, output_tokens=1))
    guard.reserve(spec, request)  # fits once the first settled at its (small) actual cost


def test_success_replaces_reservation_with_actual_cost(registry: ModelRegistry) -> None:
    guard = BudgetGuard(BudgetLimits(10, 10.0))
    spec = registry.resolve("opus")
    reservation = guard.reserve(spec, simple_request("opus"))
    actual = reservation.settle_success(Usage(input_tokens=1_000, output_tokens=100))
    snap = guard.snapshot()
    assert actual == pytest.approx((1_000 * 4.0 + 100 * 20.0) / 1e6)
    assert snap.settled_cost_usd == pytest.approx(actual)
    assert snap.outstanding_reservations_usd == pytest.approx(0.0)
    assert snap.uncertain_cost_usd == 0.0


def test_timeout_keeps_reservation_as_uncertain_charge(registry: ModelRegistry) -> None:
    guard = BudgetGuard(BudgetLimits(10, 10.0))
    reservation = guard.reserve(registry.resolve("opus"), simple_request("opus"))
    uncertain = reservation.settle_error(ProviderTimeoutError("t", provider="anthropic"))
    snap = guard.snapshot()
    assert uncertain == pytest.approx(reservation.reserved_usd)
    assert uncertain > 0
    assert snap.uncertain_cost_usd == pytest.approx(uncertain)
    assert snap.committed_cost_usd == pytest.approx(uncertain)


def test_error_status_releases_reservation_but_counts_the_attempt(registry: ModelRegistry) -> None:
    guard = BudgetGuard(BudgetLimits(10, 10.0))
    reservation = guard.reserve(registry.resolve("opus"), simple_request("opus"))
    assert reservation.settle_error(ProviderBadRequestError("x", provider="anthropic")) == 0.0
    snap = guard.snapshot()
    assert snap.committed_cost_usd == 0.0
    assert snap.attempts_dispatched == 1


def test_reservation_cannot_settle_twice(registry: ModelRegistry) -> None:
    guard = BudgetGuard(BudgetLimits(10, 10.0))
    reservation = guard.reserve(registry.resolve("opus"), simple_request("opus"))
    reservation.settle_success(Usage(input_tokens=1, output_tokens=1))
    with pytest.raises(RuntimeError, match="twice"):
        reservation.settle_success(Usage(input_tokens=1, output_tokens=1))


def test_guard_refuses_model_without_pricing(registry: ModelRegistry) -> None:
    guard = BudgetGuard(BudgetLimits(10, 10.0))
    spec = registry.resolve("opus").model_copy(update={"pricing": None})
    with pytest.raises(BudgetConfigError):
        guard.reserve(spec, simple_request("opus"))
    assert guard.snapshot().attempts_dispatched == 0
