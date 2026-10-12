from __future__ import annotations

import pytest

from apps.core.errors import BudgetConfigError
from apps.core.models import ModelRegistry, Pricing
from apps.core.pricing import (
    cost_of_usage,
    input_tokens_upper_bound,
    require_pricing,
    worst_case_attempt_cost,
)
from apps.core.types import CompletionRequest, Message, ToolSpec, Usage

# Rates chosen for easy arithmetic; not real prices.
RATES = Pricing(input=2.0, output=10.0, cache_write_5m=2.5, cache_read=0.2)


def test_cost_of_usage_known_value() -> None:
    usage = Usage(input_tokens=1_000, output_tokens=500)
    # 1000 * 2 / 1e6 + 500 * 10 / 1e6
    assert cost_of_usage(RATES, usage) == pytest.approx(0.002 + 0.005)


def test_cost_includes_reported_cache_tokens() -> None:
    usage = Usage(
        input_tokens=100,
        output_tokens=0,
        cache_read_input_tokens=1_000,
        cache_creation_input_tokens=400,
    )
    expected = (100 * 2.0 + 1_000 * 0.2 + 400 * 2.5) / 1_000_000
    assert cost_of_usage(RATES, usage) == pytest.approx(expected)


def test_long_context_rates_apply_above_threshold(registry: ModelRegistry) -> None:
    pricing = require_pricing(registry.resolve("haiku"))
    assert pricing.long_context is not None
    threshold = pricing.long_context.threshold_input_tokens
    at = Usage(input_tokens=threshold, output_tokens=0)
    above = Usage(input_tokens=threshold + 1, output_tokens=0)
    assert cost_of_usage(pricing, at) == pytest.approx(threshold * pricing.input / 1e6)
    assert cost_of_usage(pricing, above) == pytest.approx(
        (threshold + 1) * pricing.long_context.input / 1e6
    )


def test_long_context_threshold_counts_cache_tokens(registry: ModelRegistry) -> None:
    pricing = require_pricing(registry.resolve("haiku"))
    assert pricing.long_context is not None
    usage = Usage(input_tokens=10, output_tokens=0, cache_read_input_tokens=100_000)
    lc = pricing.long_context
    assert cost_of_usage(pricing, usage) == pytest.approx(
        (10 * lc.input + 100_000 * lc.cache_read) / 1e6
    )


def test_missing_pricing_blocks(registry: ModelRegistry) -> None:
    spec = registry.resolve("opus").model_copy(update={"pricing": None})
    with pytest.raises(BudgetConfigError, match="no pricing"):
        require_pricing(spec)


def test_input_upper_bound_covers_utf8_bytes() -> None:
    text = "§11.10(e) audit trail — ü" * 20
    request = CompletionRequest(
        model="fake", system="sys", messages=(Message.user(text),), max_tokens=10
    )
    assert input_tokens_upper_bound(request) >= len(text.encode("utf-8")) + len(b"sys")


def test_input_upper_bound_adds_tool_allowance() -> None:
    base = CompletionRequest(model="fake", messages=(Message.user("hi"),), max_tokens=10)
    tool = ToolSpec(name="t", description="d", input_schema={"type": "object"})
    with_tools = base.model_copy(update={"tools": (tool,)})
    assert input_tokens_upper_bound(with_tools) - input_tokens_upper_bound(base) >= 512


def test_worst_case_uses_highest_input_rate() -> None:
    # cache_write (2.5) > input (2.0), so the worst case uses 2.5 for input tokens.
    assert worst_case_attempt_cost(RATES, 1_000, 100) == pytest.approx(
        (1_000 * 2.5 + 100 * 10.0) / 1e6
    )
