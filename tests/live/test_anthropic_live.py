"""One real API call. Skipped unless --run-live is passed AND ANTHROPIC_API_KEY is set.

Cost: a single attempt with max_tokens=64, capped by a BudgetGuard (1 attempt,
$0.05 estimated). Not run in ordinary CI.
"""

from __future__ import annotations

from apps.core.budget import BudgetGuard, BudgetLimits
from apps.core.factory import build_model_client
from apps.core.models import ModelRegistry
from apps.core.settings import Settings
from apps.core.tracing.local import RecordingTracer
from tests.helpers import simple_request


def test_single_real_completion(registry: ModelRegistry) -> None:
    guard = BudgetGuard(BudgetLimits(max_request_attempts=1, max_estimated_cost_usd=0.05))
    tracer = RecordingTracer()
    client = build_model_client(
        Settings(), registry=registry, tracer=tracer, allow_paid=True, guard=guard
    )
    response = client.complete(
        simple_request("haiku", max_tokens=64, text="Reply with the single word: pong")
    )
    assert response.stop_reason in {"end_turn", "max_tokens"}
    assert response.generation.usage.input_tokens > 0
    assert response.generation.usage.output_tokens > 0
    assert response.served_model_id.startswith(response.resolved_model_id)
    assert guard.snapshot().attempts_dispatched == 1
