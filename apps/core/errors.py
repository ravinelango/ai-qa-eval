"""Error taxonomy for model calls.

Providers translate SDK exceptions into these types so callers never depend on a
specific SDK. Every ProviderError says whether a retry is reasonable and whether
the attempt may have been billed. A failed call never carries token usage.
"""

from __future__ import annotations

from typing import Literal

from apps.core.types import FailedAttempt

# "none": the provider answered with an error status, so no generation happened.
# "uncertain": we lost the connection or timed out after sending; the provider may
#              have processed (and billed) the request.
ChargeStatus = Literal["none", "uncertain"]


class QAEvalError(Exception):
    """Base class for errors raised by this repo.

    When a model call fails, ModelClient attaches the attempts made before the
    failure (to any escaping exception), so saved evidence shows the full history.
    """

    failed_attempts: tuple[FailedAttempt, ...] = ()


class ConfigurationError(QAEvalError):
    """Invalid or missing configuration. Never retried."""


class PaidCallsNotAllowedError(QAEvalError):
    """A paid provider was requested without an explicit opt-in."""


class BudgetError(QAEvalError):
    """Base for spend-control refusals."""


class BudgetConfigError(BudgetError):
    """Limits or pricing are missing, so paid execution cannot be authorized."""


class PlanRejectedError(BudgetError):
    """A planned run cannot fit within its limits, or approval was declined."""


class BudgetExceededError(BudgetError):
    """Dispatching the next attempt would exceed an approved limit."""


class ProviderError(QAEvalError):
    retryable: bool = False
    charge: ChargeStatus = "none"

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        request_id: str | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id
        self.retry_after_s = retry_after_s
        self.attempts: int = 1

    def describe(self) -> dict[str, object]:
        """Serializable summary for saved run files and traces."""
        return {
            "error_type": type(self).__name__,
            "message": str(self),
            "provider": self.provider,
            "status_code": self.status_code,
            "request_id": self.request_id,
            "retryable": self.retryable,
            "charge": self.charge,
            "attempts": self.attempts,
        }

    def to_failed_attempt(self, attempt: int, uncertain_cost_usd: float) -> FailedAttempt:
        return FailedAttempt(
            attempt=attempt,
            error_type=type(self).__name__,
            message=str(self),
            status_code=self.status_code,
            request_id=self.request_id,
            retryable=self.retryable,
            charge=self.charge,
            uncertain_cost_usd=uncertain_cost_usd,
        )


class ProviderTimeoutError(ProviderError):
    retryable = True
    charge = "uncertain"


class ProviderConnectionError(ProviderError):
    retryable = True
    charge = "uncertain"


class ProviderRateLimitError(ProviderError):
    retryable = True


class ProviderOverloadedError(ProviderError):
    retryable = True


class ProviderServerError(ProviderError):
    retryable = True


class ProviderAuthError(ProviderError):
    """401/403: bad or missing credentials, or no access."""


class ProviderBadRequestError(ProviderError):
    """4xx other than auth and rate limits, including unknown model IDs."""


class ProviderResponseError(ProviderError):
    """The provider returned 200 but the body broke our contract (e.g. no usage)."""

    charge = "uncertain"
