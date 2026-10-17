"""Shared fixtures and the opt-ins for live and model tests.

Default runs execute unit and offline integration tests with sockets disabled
(pytest-socket), so the suite cannot reach the network by accident. Live tests
call a paid API and run only with --run-live; an API key alone never enables them.

The default suite never loads a real local model (the embedder or the reranker).
Before any test or fixture runs, pytest_configure sets the flag that makes the model
loader raise; only --run-model lifts it, for the tests in tests/model/ (no network;
they need the pinned model files from `make model-fetch`).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from apps.core.models import DEFAULT_MODELS_PATH, ModelRegistry
from apps.rag import embedding


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests marked 'live' (paid API calls; also needs ANTHROPIC_API_KEY)",
    )
    parser.addoption(
        "--run-model",
        action="store_true",
        default=False,
        help="run tests marked 'model' (load the pinned local models; no network)",
    )


def pytest_configure(config: pytest.Config) -> None:
    if not config.getoption("--run-model"):
        os.environ[embedding.FORBID_MODEL_LOAD_ENV] = "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_live = config.getoption("--run-live")
    run_model = config.getoption("--run-model")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    for item in items:
        parts = Path(str(item.fspath)).parts
        for marker in ("unit", "integration", "live", "model"):
            if marker in parts:
                item.add_marker(getattr(pytest.mark, marker))
        if "model" in parts and not run_model:
            item.add_marker(pytest.mark.skip(reason="model test: pass --run-model to run it"))
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
