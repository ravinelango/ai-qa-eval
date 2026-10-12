"""Shared fixtures and the live-test opt-in.

Default runs execute unit and offline integration tests with sockets disabled
(pytest-socket), so the suite cannot reach the network by accident. Live tests
call a paid API and run only with --run-live; an API key alone never enables them.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from apps.core.models import DEFAULT_MODELS_PATH, ModelRegistry


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests marked 'live' (paid API calls; also needs ANTHROPIC_API_KEY)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_live = config.getoption("--run-live")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    for item in items:
        parts = Path(str(item.fspath)).parts
        for marker in ("unit", "integration", "live"):
            if marker in parts:
                item.add_marker(getattr(pytest.mark, marker))
        if "live" not in parts:
            continue
        if not run_live:
            item.add_marker(pytest.mark.skip(reason="live test: pass --run-live to run it"))
        elif not has_key:
            item.add_marker(pytest.mark.skip(reason="live test: ANTHROPIC_API_KEY is not set"))
        else:
            item.add_marker(pytest.mark.enable_socket)


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.load()


@pytest.fixture
def models_yaml() -> dict[str, object]:
    """The shipped config as a dict, for tests that write variants of it."""
    data: dict[str, object] = yaml.safe_load(DEFAULT_MODELS_PATH.read_text(encoding="utf-8"))
    return data


@pytest.fixture
def write_models(tmp_path: Path) -> Callable[[dict[str, object]], Path]:
    def _write(data: dict[str, object]) -> Path:
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write
