"""Builds tracers and model clients from settings. Imports SDKs only when needed."""

from __future__ import annotations

from apps.core.budget import BudgetGuard
from apps.core.client import ModelClient
from apps.core.errors import ConfigurationError
from apps.core.models import ModelRegistry
from apps.core.providers.base import ProviderBackend
from apps.core.settings import Settings
from apps.core.tracing.base import Tracer
from apps.core.tracing.local import NoopTracer, RecordingTracer


def build_tracer(settings: Settings, *, run_id: str | None = None) -> Tracer:
    if settings.tracing_backend == "none":
        return NoopTracer()
    if settings.tracing_backend == "local":
        name = f"{run_id}.jsonl" if run_id else "traces.jsonl"
        return RecordingTracer(path=settings.trace_dir / name)
    public, secret = settings.langfuse_public_key, settings.langfuse_secret_key
    if public is None or secret is None:
        raise ConfigurationError(
            "TRACING_BACKEND=langfuse needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
        )
    from langfuse import Langfuse

    from apps.core.tracing.langfuse_tracer import LangfuseTracer

    client = Langfuse(
        public_key=public.get_secret_value(),
        secret_key=secret.get_secret_value(),
        base_url=settings.langfuse_base_url,
    )
    return LangfuseTracer(client)


def build_model_client(
    settings: Settings,
    *,
    registry: ModelRegistry,
    tracer: Tracer,
    allow_paid: bool,
    guard: BudgetGuard | None,
) -> ModelClient:
    """A client for real providers. Without allow_paid no paid backend is even built."""
    backends: dict[str, ProviderBackend] = {}
    if allow_paid:
        if settings.anthropic_api_key is None:
            raise ConfigurationError("--allow-paid was given but ANTHROPIC_API_KEY is not set")
        from apps.core.providers.anthropic import AnthropicBackend

        backends["anthropic"] = AnthropicBackend(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout_s=registry.provider_settings("anthropic").timeout_s,
        )
    return ModelClient(
        registry=registry, backends=backends, tracer=tracer, allow_paid=allow_paid, guard=guard
    )
