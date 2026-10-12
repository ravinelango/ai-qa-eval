"""Keep credentials out of logs and saved evidence.

Two layers, both installed by `configure_logging`:
- `RedactingFilter` sanitizes the record itself (message, arguments, formatted
  exception text including chained causes, stack info), so every handler that
  sees the record afterwards gets the clean version.
- `RedactingFormatter` redacts the complete formatted output as a final check.

Configured secret values are masked, as is anything shaped like an Anthropic or
Langfuse key.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable
from typing import TextIO

MASK = "[REDACTED]"
_KEY_PATTERNS = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{8,}|sk-lf-[A-Za-z0-9_\-]{8,}|pk-lf-[A-Za-z0-9_\-]{8,})"
)
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, MASK)
    return _KEY_PATTERNS.sub(MASK, text)


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secrets)
        self._exception_formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage(), self._secrets)
        record.args = None
        if record.exc_info:
            # formatException renders the whole chain (__cause__ / __context__).
            text = record.exc_text or self._exception_formatter.formatException(record.exc_info)
            record.exc_text = redact(text, self._secrets)
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact(record.exc_text, self._secrets)
        if record.stack_info:
            record.stack_info = redact(record.stack_info, self._secrets)
        return True


class RedactingFormatter(logging.Formatter):
    def __init__(self, fmt: str | None = None, *, secrets: Iterable[str] = ()) -> None:
        super().__init__(fmt)
        self._secrets = tuple(secrets)

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record), self._secrets)


def configure_logging(
    secrets: Iterable[str] = (),
    level: int = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Handler:
    """Replace the root handlers with one redacting handler. Call from every entry point."""
    secrets = tuple(secrets)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.addFilter(RedactingFilter(secrets))
    handler.setFormatter(RedactingFormatter(_DEFAULT_FORMAT, secrets=secrets))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    return handler
