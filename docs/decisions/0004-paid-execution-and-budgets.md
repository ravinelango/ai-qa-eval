# 0004 — Paid execution and spending limits

**Status:** accepted (Phase 0) · **Date:** 2026-10-12

## Decision
- **Opt-in.** An API key alone enables nothing. Paid calls need `--allow-paid`
  (`apps/core/paid_cli.py`) and a `BudgetGuard`; `ModelClient` refuses paid aliases
  otherwise. Live tests need `--run-live` *and* a key.
- **Plans count everything.** `PlannedCalls` lines cover generation, each judge metric and
  every repeat; the estimate reports attempts and cost with no retries and with all retries.
- **Interactive runs**: if the worst-case attempt count exceeds
  `approval_threshold_attempts` (50, in `config/budgets.yaml`), the estimate is printed and
  the run waits for `yes`. The approved worst case becomes the run's limit.
- **Run mode is detected, not assumed.** A run is interactive only when `--unattended`
  is absent, no CI marker is set (`CI`, `GITHUB_ACTIONS`, `GITLAB_CI`, `BUILDKITE`,
  `JENKINS_URL`, `TEAMCITY_VERSION`, `TF_BUILD`; values `false`/`0` ignored) and stdin is
  a terminal. Anything else is unattended, so forgetting `--unattended` in CI cannot fall
  back to the interactive path.
- **Unattended runs (CI)** never prompt. They name a profile in `config/budgets.yaml`.
  Missing or null limits stop the run; so does a plan that cannot fit even without
  retries. The shipped profiles are null on purpose: the repo owner sets them.
- **Per attempt**: before dispatch the guard reserves a worst-case cost (input upper bound
  from UTF-8 byte length plus allowances, priced at the highest input rate, plus
  `max_tokens` output) and refuses the attempt if attempts or committed cost would exceed
  the limit. Success replaces the reservation with the cost from reported usage. A timeout,
  dropped connection, malformed 200 or unexpected exception keeps the reservation as an
  **uncertain** charge. An error status releases it but still counts the attempt. Every
  dispatched attempt ends settled; saved budget records also show
  `outstanding_reservations_usd` (non-zero means an attempt never settled) and
  `committed_cost_usd`.
- **Pricing** comes from the dated table in `config/models.yaml`. A paid model without
  pricing cannot be estimated or reserved.

## What the limit means
The monetary limit caps *estimated* spend from the pricing table. It is not the provider's
bill; the Anthropic console is the only authoritative record. Phase 0 makes no paid calls.
