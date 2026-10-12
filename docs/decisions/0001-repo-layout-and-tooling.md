# 0001 — Repo layout, packaging and tooling

**Status:** accepted (Phase 0) · **Date:** 2026-10-12

## Decision
- One uv project, Python 3.12 (`requires-python = ">=3.12,<3.13"`, `.python-version`).
- Top-level packages `apps`, `evals`, `redteam`, `dashboard`, listed explicitly in
  `[tool.hatch.build.targets.wheel]`; nothing is discovered implicitly. Shared code lives
  in `apps/core/`. CI builds the wheel and `scripts/check_wheel.py` fails if a package is missing.
- Exact `==` pins in `pyproject.toml`; `uv.lock` is committed and CI installs with
  `uv sync --locked`.
- ruff (lint + format), mypy `--strict`, pytest. pre-commit runs ruff/mypy through
  `uv run`, so hook versions come from the lockfile.
- The default test run disables sockets (pytest-socket). A test that touches the network fails.

## Dependency groups are not isolation
uv resolves all dependency groups into one lockfile unless conflicts are declared
explicitly. Groups keep installs smaller; they do not let incompatible packages coexist.
NVIDIA garak will get its own environment in Phase 2 as a precaution. No dependency
conflict with Ragas/DeepEval has been reproduced; if one is found, it will be recorded here
with the failing resolution output.

## Consequences
The project runs from a checkout (uv installs it in editable mode). `config/` is read from
the repo root and is not shipped in the wheel; the wheel build only verifies packaging.
Each phase adds its own pinned dependencies. Generic package names (`apps`, `evals`)
could collide with other distributions in a shared environment; we only install into the
project's own `.venv`.
