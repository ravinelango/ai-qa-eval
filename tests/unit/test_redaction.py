"""Regression tests for review finding 2: secrets in exception tracebacks."""

from __future__ import annotations

import io
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from apps.core import demo
from apps.core.client import ModelClient
from apps.core.errors import ProviderBadRequestError
from apps.core.models import ModelRegistry
from apps.core.providers.fake import FakeBackend
from apps.core.redact import MASK, RedactingFilter, RedactingFormatter, configure_logging
from tests.helpers import simple_request

SECRET = "synthetic-secret-value-123"  # not a real credential


@pytest.fixture
def restore_root_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


def _raise_chained() -> None:
    try:
        raise ValueError(f"inner detail {SECRET}")
    except ValueError as inner:
        raise RuntimeError(f"outer detail {SECRET}") from inner


def test_logging_exception_masks_secret_in_chained_traceback(restore_root_logging: None) -> None:
    stream = io.StringIO()
    configure_logging([SECRET], stream=stream)
    try:
        _raise_chained()
    except RuntimeError:
        logging.getLogger("tests.redaction").exception("call failed")
    output = stream.getvalue()
    assert SECRET not in output
    assert "inner detail" in output  # the cause is still rendered...
    assert "outer detail" in output
    assert output.count(MASK) >= 2  # ...with both occurrences masked


def test_stack_info_is_masked(restore_root_logging: None) -> None:
    stream = io.StringIO()
    configure_logging([SECRET], stream=stream)
    logging.getLogger("tests.redaction").warning("value %s", SECRET, stack_info=True)
    assert SECRET not in stream.getvalue()
    assert "Stack (most recent call last)" in stream.getvalue()


def test_later_handlers_receive_the_sanitized_record(restore_root_logging: None) -> None:
    configure_logging([SECRET], stream=io.StringIO())
    plain_stream = io.StringIO()
    plain = logging.StreamHandler(plain_stream)  # no redaction of its own
    logging.getLogger().addHandler(plain)
    try:
        _raise_chained()
    except RuntimeError:
        logging.getLogger("tests.redaction").exception("call failed")
    assert SECRET not in plain_stream.getvalue()
    assert "inner detail" in plain_stream.getvalue()


def test_formatter_alone_masks_complete_output() -> None:
    formatter = RedactingFormatter("%(message)s", secrets=[SECRET])
    try:
        _raise_chained()
    except RuntimeError:
        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "msg", None, sys.exc_info())
    assert SECRET not in formatter.format(record)


def test_filter_handles_preformatted_exc_text() -> None:
    record = logging.LogRecord("t", logging.ERROR, __file__, 1, "msg", None, None)
    record.exc_text = f"Traceback ... {SECRET}"
    RedactingFilter([SECRET]).filter(record)
    assert record.exc_text is not None
    assert SECRET not in record.exc_text


def test_provider_failure_logged_by_client_is_masked(
    restore_root_logging: None, registry: ModelRegistry
) -> None:
    stream = io.StringIO()
    configure_logging([SECRET], stream=stream)
    error = ProviderBadRequestError(f"HTTP 400: echoed {SECRET}", provider="fake", status_code=400)
    client = ModelClient(registry=registry, backends={"fake": FakeBackend([error])})
    with pytest.raises(ProviderBadRequestError):
        client.complete(simple_request("fake"))
    output = stream.getvalue()
    assert "model call attempt 1 failed" in output
    assert SECRET not in output


def test_demo_entry_point_installs_redacting_logging(
    restore_root_logging: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    demo.main(["--out", str(tmp_path)])
    (handler,) = logging.getLogger().handlers
    assert any(isinstance(f, RedactingFilter) for f in handler.filters)
    assert isinstance(handler.formatter, RedactingFormatter)
    logging.getLogger("tests.redaction").warning("key is %s", SECRET)
    err = capsys.readouterr().err
    assert "model call attempt 1 failed (ProviderRateLimitError, retrying)" in err
    assert SECRET not in err
