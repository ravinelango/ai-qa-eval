"""Cost estimation from the dated pricing table in config/models.yaml.

These are estimates. The provider's invoice is the only authoritative bill.
"""

from __future__ import annotations

from apps.core.errors import BudgetConfigError
from apps.core.models import ModelSpec, Pricing
from apps.core.types import CompletionRequest, Usage

_PER_MTOK = 1_000_000

# Conservative allowances for tokens the provider adds on top of what we send:
# role/formatting tokens per message and the tool-use system prompt. The pricing
# page lists 286 tool-use system prompt tokens for current models; we reserve more.
_MESSAGE_OVERHEAD_TOKENS = 16
_REQUEST_OVERHEAD_TOKENS = 64
_TOOL_SYSTEM_PROMPT_ALLOWANCE = 512


def require_pricing(spec: ModelSpec) -> Pricing:
    if spec.pricing is None:
        raise BudgetConfigError(
            f"model alias {spec.alias!r} has no pricing in config/models.yaml; "
            "paid execution is blocked until it is added"
        )
    return spec.pricing


def cost_of_usage(pricing: Pricing, usage: Usage) -> float:
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    rates = pricing.rates_for(usage.input_tokens + cache_read + cache_write)
    return (
        usage.input_tokens * rates.input
        + usage.output_tokens * rates.output
        + cache_read * rates.cache_read
        + cache_write * rates.cache_write_5m
    ) / _PER_MTOK


def input_tokens_upper_bound(request: CompletionRequest) -> int:
    """An upper bound on billed input tokens, computed without a network call.

    Byte-level BPE tokens encode at least one byte each, so the UTF-8 byte length
    of the content bounds its token count; fixed allowances cover formatting and
    the tool-use system prompt. Opaque blocks are counted by their JSON size.
    """
    total = _REQUEST_OVERHEAD_TOKENS
    if request.system:
        total += len(request.system.encode("utf-8"))
    for message in request.messages:
        total += _MESSAGE_OVERHEAD_TOKENS
        for block in message.content:
            total += len(block.model_dump_json().encode("utf-8"))
    if request.tools:
        total += _TOOL_SYSTEM_PROMPT_ALLOWANCE
        for tool in request.tools:
            total += len(tool.model_dump_json().encode("utf-8"))
    return total


def worst_case_attempt_cost(
    pricing: Pricing, max_input_tokens: int, max_output_tokens: int
) -> float:
    """Cost of one attempt if it used its whole input bound and output limit.

    Uses the most expensive applicable input rate (cache writes cost more than
    plain input), so the reservation is never lower than the real charge.
    """
    rates = pricing.rates_for(max_input_tokens)
    input_rate = max(rates.input, rates.cache_write_5m)
    return (max_input_tokens * input_rate + max_output_tokens * rates.output) / _PER_MTOK
