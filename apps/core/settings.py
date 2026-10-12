"""Runtime settings, read from environment variables (or a local, untracked .env).

Secrets are SecretStr so they never print in reprs or logs. Having a key set
enables nothing by itself: paid calls also need an explicit --allow-paid and
approved budget limits (apps.core.budget).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from apps.core.models import REPO_ROOT

TracingBackend = Literal["none", "local", "langfuse"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    tracing_backend: TracingBackend = "none"
    trace_dir: Path = REPO_ROOT / "runs" / "traces"

    def secret_values(self) -> list[str]:
        secrets = (self.anthropic_api_key, self.langfuse_public_key, self.langfuse_secret_key)
        return [s.get_secret_value() for s in secrets if s is not None and s.get_secret_value()]
