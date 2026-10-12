# ai-qa-eval: LLM quality engineering portfolio

A monorepo with three projects that test an LLM application the way a QA team would:

1. **RAG evaluation harness with a CI release gate**: golden dataset, Ragas and DeepEval
   metrics, a validated LLM judge, and a gate that blocks regressions. *(Phase 1)*
2. **Agent red-team suite**: hand-authored attacks plus promptfoo and garak scans
   against this repo's own agent, mapped to the OWASP Top 10 for LLM Applications. *(Phase 2)*
3. **Model-upgrade regression dashboard**: paired statistics and go/no-go rules for
   swapping models or prompts. *(Phase 3)*

The corpus is FDA 21 CFR Part 11 and related public FDA guidance.

## Ground rules

- **No fabricated evidence.** Every number in a report comes from a saved run. Anything
  not run is marked "not yet run". Fake-provider output is labeled synthetic.
- **Paid calls are opt-in.** An API key alone enables nothing. Paid commands need
  `--allow-paid` and approved budget limits ([0004](docs/decisions/0004-paid-execution-and-budgets.md)).
- **Secrets come from environment variables** (see `.env.example`) and are redacted from logs.
- **Red-team targets only the local agent in this repo.**

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, provider contract, budget guard, tracing, CI | **done, in review** |
| 1 | RAG assistant, golden dataset, metrics, judge validation, CI gate | not started |
| 2 | Agent, adversarial cases, promptfoo/garak, findings report | not started |
| 3 | Experiment runner, stats module, Streamlit dashboard | not started |

No evaluation, judge-validation, red-team or A/B results exist yet. `reports/` is empty
on purpose.

## Quickstart (no API keys, no network at runtime)

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 if needed.

```bash
make setup      # uv sync --locked + pre-commit hooks
make check      # ruff, mypy --strict, unit + offline integration tests
make demo       # scripted fake-provider demo -> runs/demo/<run_id>/{run.json,trace.jsonl}
```

`make demo` prints a SYNTHETIC banner. Its "model replies" are fixed strings in
`apps/core/demo.py`. It shows the provider contract end to end: a structured tool call,
the tool result sent back, a final answer, a rate-limit error retried by our own retry
loop, and one trace ID shared by the run file and the trace file.

`make help` lists every target. Targets for later phases exit with
`not yet implemented (Phase N)`.

## Repo layout

```
apps/core/         Provider contract, model registry, budget guard, tracing, evidence
apps/rag/          RAG assistant (Phase 1a)
apps/agent/        Tool-using agent, red-team target (Phase 2a)
evals/             Golden dataset, metrics, judge validation, gate logic (Phase 1)
redteam/           Adversarial cases, promptfoo/garak configs, findings (Phase 2)
dashboard/         Streamlit dashboard and stats module (Phase 3)
config/            models.yaml (model IDs + dated pricing), budgets.yaml (spend limits)
data/corpus/       Source documents + SOURCES.md (Phase 1a)
reports/           Generated reports (none yet)
docs/decisions/    Decision records
tests/             unit/, integration/ (offline, real SDKs), live/ (paid, opt-in)
```

`ai-qa-eval.zip` at the root holds an earlier, separate project (a Playwright-log LLM
judge). It is left untouched for now. It will be extracted into browsable source in its
own PR, with its tests and live mode preserved.

## Phase 0: scaffold

**What it provides**
- **Provider contract** ([0002](docs/decisions/0002-provider-contract.md)):
  - `ModelClient` resolves model aliases from `config/models.yaml`.
  - It supports structured and multiple tool calls, tool results, and opaque provider
    blocks (thinking) that are round-tripped unchanged.
  - Stop reasons, refusal categories and token usage are preserved. Usage is never
    invented on failure.
  - Errors have an explicit taxonomy with retry and charge semantics.
- **Anthropic backend** on the official SDK, with SDK retries off. Our retry loop sends
  every attempt through the budget guard.
- **Budget guard** ([0004](docs/decisions/0004-paid-execution-and-budgets.md)):
  - Plan estimates count generation, judging, repeats and retries.
  - Interactive runs over 50 attempts need approval. Unattended runs need configured
    limits.
  - Each attempt reserves budget before it is dispatched. Timeouts are recorded as
    uncertain charges.
- **Cache semantics** ([0003](docs/decisions/0003-response-cache.md)): the key function
  and policy are fixed now. The disk cache comes in Phase 1e.
- **Tracing** ([0005](docs/decisions/0005-tracing-and-evidence.md)): the backends are
  none, local JSONL and Langfuse. Run files are the primary evidence, and trace IDs
  link to them.
- **CI** (`.github/workflows/ci.yml`): lint, format, types, tests, the demo and a wheel
  package check. It uses no secrets.

**How it is verified**
- **Unit tests** use the scripted `FakeBackend`.
- **Offline integration tests** run the real `anthropic` SDK against an in-process mock
  HTTP transport, and the real Langfuse SDK against an in-memory OpenTelemetry exporter.
- **No network:** the whole default suite runs with sockets disabled.
- **Live test:** `tests/live/` makes one paid call. It runs only with `make test-live`.

**Not yet verified live**
- No call has been made to the Anthropic API.
- No trace has been sent to Langfuse Cloud.
- Both integrations are tested against their SDKs offline only.

## Limitations (Phase 0)

- The cost limit caps *estimated* spend from the dated pricing table. It is not the
  provider's bill.
- The input-token upper bound is a byte-length heuristic. It deliberately over-reserves.
- The "no charge on error status" assumption ([0002](docs/decisions/0002-provider-contract.md))
  has not been confirmed against an invoice.
