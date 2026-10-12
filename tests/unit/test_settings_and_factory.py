from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pytest

from apps.core.errors import ConfigurationError, PaidCallsNotAllowedError
from apps.core.factory import build_model_client, build_tracer
from apps.core.models import ModelRegistry
from apps.core.redact import MASK, RedactingFilter, redact
from apps.core.settings import Settings
from apps.core.tracing.local import NoopTracer, RecordingTracer
from tests.helpers import simple_request

# Synthetic values shaped like credentials, for redaction tests. Not real keys.
FAKE_KEY = "sk-ant-api03-FAKEFAKEFAKEFAKE"
FAKE_LF_KEY = "sk-lf-0123456789abcdef"  # gitleaks:allow


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "TRACING_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_defaults_need_no_keys(clean_env: None) -> None:
    settings = _settings()
    assert settings.anthropic_api_key is None
    assert settings.tracing_backend == "none"


def test_secrets_never_appear_in_repr(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    settings = _settings()
    assert FAKE_KEY not in repr(settings)
    assert settings.secret_values() == [FAKE_KEY]


def test_key_in_environment_does_not_enable_paid_calls(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, registry: ModelRegistry
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    client = build_model_client(
        _settings(), registry=registry, tracer=NoopTracer(), allow_paid=False, guard=None
    )
    with pytest.raises(PaidCallsNotAllowedError):
        client.complete(simple_request("opus"))


def test_allow_paid_without_key_fails_fast(clean_env: None, registry: ModelRegistry) -> None:
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        build_model_client(
            _settings(), registry=registry, tracer=NoopTracer(), allow_paid=True, guard=None
        )


def test_tracer_backends(clean_env: None, tmp_path: Path) -> None:
    assert isinstance(build_tracer(_settings()), NoopTracer)
    local = build_tracer(_settings(tracing_backend="local", trace_dir=tmp_path), run_id="r1")
    assert isinstance(local, RecordingTracer)
    assert local.path == tmp_path / "r1.jsonl"
    with pytest.raises(ConfigurationError, match="LANGFUSE_PUBLIC_KEY"):
        build_tracer(_settings(tracing_backend="langfuse"))


def test_redact_masks_configured_secrets_and_key_shapes() -> None:
    text = f"key={FAKE_KEY} other=hunter2-secret lf={FAKE_LF_KEY}"
    out = redact(text, ["hunter2-secret"])
    assert FAKE_KEY not in out
    assert "hunter2-secret" not in out
    assert FAKE_LF_KEY not in out
    assert out.count(MASK) == 3


def test_logging_filter_redacts_formatted_messages() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter(["my-secret-value"]))
    logger = logging.getLogger("tests.redaction")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.warning("calling with %s and %s", "my-secret-value", FAKE_KEY)
    finally:
        logger.removeHandler(handler)
    output = stream.getvalue()
    assert "my-secret-value" not in output
    assert FAKE_KEY not in output
    assert MASK in output
