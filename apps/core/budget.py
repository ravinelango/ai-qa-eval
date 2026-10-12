"""Spend control for paid model calls (docs/decisions/0004-paid-execution.md).

Three pieces:
1. `estimate_plan` turns a run plan (generation, judging, repeats) into attempt and
   cost estimates, with and without retries.
2. `authorize_run` decides the limits for a run: interactive runs above the
   approval threshold need an explicit "yes"; unattended runs use limits from
   config/budgets.yaml and stop if those are missing.
3. `BudgetGuard` enforces the limits per attempt: it reserves a worst-case cost
   before dispatch and settles it afterwards. Timeouts keep their reservation as
   an uncertain charge instead of being treated as free.

All amounts are estimates from the dated pricing table, not the provider's bill.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apps.core.errors import (
    BudgetConfigError,
    BudgetExceededError,
    ConfigurationError,
    PlanRejectedError,
    ProviderError,
)
from apps.core.models import REPO_ROOT, ModelRegistry, ModelSpec, Pricing
from apps.core.pricing import (
    cost_of_usage,
    input_tokens_upper_bound,
    require_pricing,
    worst_case_attempt_cost,
)
from apps.core.types import CompletionRequest, Usage

DEFAULT_BUDGETS_PATH = REPO_ROOT / "config" / "budgets.yaml"

RunMode = Literal["interactive", "unattended"]


# --- configuration ---------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProfileLimits(_Strict):
    max_request_attempts: int | None = Field(default=None, gt=0)
    max_estimated_cost_usd: float | None = Field(default=None, gt=0)


class BudgetsConfig(_Strict):
    approval_threshold_attempts: int = Field(gt=0)
    profiles: dict[str, ProfileLimits]

    @classmethod
    def load(cls, path: Path = DEFAULT_BUDGETS_PATH) -> BudgetsConfig:
        try:
            return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (OSError, ValidationError) as exc:
            raise ConfigurationError(f"invalid budget config {path}: {exc}") from exc


@dataclass(frozen=True)
class BudgetLimits:
    max_request_attempts: int
    max_estimated_cost_usd: float


# --- planning ----------------------------------------------------------------------


class PlannedCalls(_Strict):
    """One line of a run plan. `calls` already includes repeats (e.g. 20 cases x 3 runs)."""

    purpose: str
    model_alias: str
    calls: int = Field(ge=0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)


@dataclass(frozen=True)
class PlanLine:
    item: PlannedCalls
    attempts_no_retry: int
    attempts_all_retries: int
    cost_no_retry_usd: float
    cost_all_retries_usd: float


@dataclass(frozen=True)
class SpendEstimate:
    lines: tuple[PlanLine, ...]
    pricing_checked: str

    @property
    def attempts_no_retry(self) -> int:
        return sum(line.attempts_no_retry for line in self.lines)

    @property
    def attempts_all_retries(self) -> int:
        return sum(line.attempts_all_retries for line in self.lines)

    @property
    def cost_no_retry_usd(self) -> float:
        return sum(line.cost_no_retry_usd for line in self.lines)

    @property
    def cost_all_retries_usd(self) -> float:
        return sum(line.cost_all_retries_usd for line in self.lines)

    def render(self) -> str:
        header = (
            f"{'purpose':<28}{'model':<10}{'calls':>7}{'attempts(max)':>15}"
            f"{'est. cost':>12}{'est. cost(max)':>16}"
        )
        rows = [header, "-" * len(header)]
        for ln in self.lines:
            rows.append(
                f"{ln.item.purpose:<28}{ln.item.model_alias:<10}{ln.item.calls:>7}"
                f"{ln.attempts_all_retries:>15}{ln.cost_no_retry_usd:>12.4f}"
                f"{ln.cost_all_retries_usd:>16.4f}"
            )
        rows.append("-" * len(header))
        rows.append(
            f"{'TOTAL':<38}{self.attempts_no_retry:>7}{self.attempts_all_retries:>15}"
            f"{self.cost_no_retry_usd:>12.4f}{self.cost_all_retries_usd:>16.4f}"
        )
        rows.append(
            "Costs are upper-bound estimates (full input bound and max_tokens per attempt) "
            f"from pricing checked {self.pricing_checked}; they are not the provider's bill."
        )
        return "\n".join(rows)


def estimate_plan(plan: list[PlannedCalls], registry: ModelRegistry) -> SpendEstimate:
    lines = []
    for item in plan:
        spec = registry.resolve(item.model_alias)
        settings = registry.provider_settings(spec.provider)
        per_attempt = (
            worst_case_attempt_cost(
                require_pricing(spec), item.max_input_tokens, item.max_output_tokens
            )
            if spec.is_paid
            else 0.0
        )
        attempts_max = item.calls * (1 + settings.max_retries)
        lines.append(
            PlanLine(
                item=item,
                attempts_no_retry=item.calls,
                attempts_all_retries=attempts_max,
                cost_no_retry_usd=item.calls * per_attempt,
                cost_all_retries_usd=attempts_max * per_attempt,
            )
        )
    return SpendEstimate(lines=tuple(lines), pricing_checked=registry.pricing_checked)


# --- authorization -----------------------------------------------------------------


def authorize_run(
    estimate: SpendEstimate,
    *,
    mode: RunMode,
    budgets: BudgetsConfig,
    profile: str | None = None,
    confirm: Callable[[str], bool] | None = None,
    mode_reason: str | None = None,
) -> BudgetLimits:
    """Return the limits this run must stay within, or raise.

    Interactive: the limits are the plan's own worst case (retries included). Plans
    above the approval threshold are shown to the user and need confirm() -> True.
    Unattended: never prompts. Limits come from the named profile; missing limits
    stop the run, and a plan that cannot fit even without retries is rejected.
    `mode_reason` (why the run is unattended) is included in refusals.
    """
    if mode == "interactive":
        limits = BudgetLimits(
            max_request_attempts=estimate.attempts_all_retries,
            max_estimated_cost_usd=estimate.cost_all_retries_usd,
        )
        if estimate.attempts_all_retries > budgets.approval_threshold_attempts:
            if confirm is None:
                raise PlanRejectedError(
                    "plan needs approval but no confirmation prompt is available"
                )
            question = (
                f"{estimate.render()}\n\nThis run may make up to "
                f"{estimate.attempts_all_retries} paid request attempts "
                f"(estimated <= ${estimate.cost_all_retries_usd:.4f}). Proceed?"
            )
            if not confirm(question):
                raise PlanRejectedError("run declined at the approval prompt")
        return limits

    why = f"unattended run ({mode_reason})" if mode_reason else "unattended run"
    if profile is None:
        raise BudgetConfigError(f"{why}: paid execution needs --budget-profile")
    configured = budgets.profiles.get(profile)
    if configured is None:
        raise BudgetConfigError(
            f"{why}: budget profile {profile!r} is not defined in config/budgets.yaml"
        )
    if configured.max_request_attempts is None or configured.max_estimated_cost_usd is None:
        raise BudgetConfigError(
            f"{why}: budget profile {profile!r} has no approved limits; paid execution is blocked"
        )
    limits = BudgetLimits(configured.max_request_attempts, configured.max_estimated_cost_usd)
    if estimate.attempts_no_retry > limits.max_request_attempts:
        raise PlanRejectedError(
            f"plan needs {estimate.attempts_no_retry} attempts; profile {profile!r} allows "
            f"{limits.max_request_attempts}"
        )
    if estimate.cost_no_retry_usd > limits.max_estimated_cost_usd:
        raise PlanRejectedError(
            f"plan estimate ${estimate.cost_no_retry_usd:.4f} exceeds profile {profile!r} "
            f"limit ${limits.max_estimated_cost_usd:.4f}"
        )
    return limits


# --- enforcement -------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetSnapshot:
    attempts_dispatched: int
    settled_cost_usd: float
    uncertain_cost_usd: float
    outstanding_reservations_usd: float
    limits: BudgetLimits

    @property
    def committed_cost_usd(self) -> float:
        return self.settled_cost_usd + self.uncertain_cost_usd + self.outstanding_reservations_usd


@dataclass
class Reservation:
    """A worst-case cost held for one attempt until it is settled exactly once."""

    guard: BudgetGuard
    model_alias: str
    pricing: Pricing
    reserved_usd: float
    _settled: bool = field(default=False, init=False)

    def settle_success(self, usage: Usage) -> float:
        actual = cost_of_usage(self.pricing, usage)
        self.guard._settle(self, settled=actual, uncertain=0.0)
        return actual

    def settle_error(self, error: ProviderError) -> float:
        """Returns the cost recorded as uncertain (0 when no charge occurred)."""
        uncertain = self.reserved_usd if error.charge == "uncertain" else 0.0
        self.guard._settle(self, settled=0.0, uncertain=uncertain)
        return uncertain


class BudgetGuard:
    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self._lock = threading.Lock()
        self._attempts = 0
        self._settled = 0.0
        self._uncertain = 0.0
        self._outstanding = 0.0

    def reserve(self, spec: ModelSpec, request: CompletionRequest) -> Reservation:
        """Check and reserve allowance for one attempt. Raises before dispatch."""
        pricing = require_pricing(spec)
        estimate = worst_case_attempt_cost(
            pricing, input_tokens_upper_bound(request), request.max_tokens
        )
        with self._lock:
            if self._attempts + 1 > self.limits.max_request_attempts:
                raise BudgetExceededError(
                    f"attempt limit reached ({self.limits.max_request_attempts}); "
                    "not dispatching another request"
                )
            committed = self._settled + self._uncertain + self._outstanding
            if committed + estimate > self.limits.max_estimated_cost_usd:
                raise BudgetExceededError(
                    f"next attempt could cost up to ${estimate:.4f}; committed "
                    f"${committed:.4f} of ${self.limits.max_estimated_cost_usd:.4f}"
                )
            self._attempts += 1
            self._outstanding += estimate
        return Reservation(self, spec.alias, pricing, estimate)

    def _settle(self, reservation: Reservation, *, settled: float, uncertain: float) -> None:
        with self._lock:
            if reservation._settled:
                raise RuntimeError("reservation settled twice")
            reservation._settled = True
            self._outstanding -= reservation.reserved_usd
            self._settled += settled
            self._uncertain += uncertain

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                attempts_dispatched=self._attempts,
                settled_cost_usd=self._settled,
                uncertain_cost_usd=self._uncertain,
                outstanding_reservations_usd=self._outstanding,
                limits=self.limits,
            )
