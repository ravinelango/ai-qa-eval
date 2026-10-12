"""Keep credentials out of logs.

`RedactingFilter` masks configured secret values and anything shaped like an
Anthropic or Langfuse key. Attach it to log handlers (configure_logging does).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

MASK = "[REDACTED]"
_KEY_PATTERNS = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{8,}|sk-lf-[A-Za-z0-9_\-]{8,}|pk-lf-[A-Za-z0-9_\-]{8,})"
)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, MASK)
    return _KEY_PATTERNS.sub(MASK, text)


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage(), self._secrets)
        record.args = None
        return True


def configure_logging(secrets: Iterable[str] = (), level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter(secrets))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
