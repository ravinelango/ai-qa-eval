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
  - A run is interactive only in a terminal with no CI markers. Anything else is
    unattended: it never prompts and needs an approved budget profile.
  - Interactive runs over 50 attempts need approval.
  - Each attempt reserves budget before it is dispatched, and every attempt ends
    settled. Timeouts, malformed responses and unexpected exceptions are recorded as
    uncertain charges.
- **Cache semantics** ([0003](docs/decisions/0003-response-cache.md)): the key function
  and policy are fixed now; the disk cache comes in Phase 1e. Caching gives repeatable
  replay of recorded outputs. Fresh model behavior and judge variance still need
  uncached measurements.
- **Evidence and tracing** ([0005](docs/decisions/0005-tracing-and-evidence.md)): run
  files stand alone. They hold resolved settings, requests, responses or errors, and
  every failed attempt, with any tracer or none. Tracing backends are none, local
  JSONL and Langfuse; trace IDs are supplementary links.
- **Redaction:** entry points install logging that masks secrets in messages and full
  tracebacks. Run and trace files mask credential-shaped strings.
- **CI** (`.github/workflows/ci.yml`): lint, format, types, tests, the demo and a wheel
  package check. It uses no secrets.

**How it is verified**
- **Unit tests** use the scripted `FakeBackend`.
- **Offline integration tests** run the real `anthropic` SDK against an in-process mock
  HTTP transport, and the real Langfuse SDK against an in-memory OpenTelemetry exporter.
- **No network:** the whole default suite runs with sockets disabled. Integration tests
  also run under a hostile SOCKS proxy environment, so they cannot depend on host proxy
  settings.
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
- Pattern-based redaction only catches configured secrets and Anthropic/Langfuse
  key shapes.

## Decisions recorded for Phase 1

- **Generator:** Sonnet (`sonnet` alias), provisional.
- **Judge:** provisional until validated against the owner's human labels (Phase 1d).
- **Paid CI budget profiles** stay unset until the owner approves concrete limits.
- **Live check** (`make test-live`) is deferred until the Phase 0 fixes are reviewed.
