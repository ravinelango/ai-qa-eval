# 0002 — Provider contract

**Status:** accepted (Phase 0) · **Date:** 2026-10-12

## Decision
Application code calls `ModelClient.complete(CompletionRequest) -> CompletionResponse`
(`apps/core/client.py`). Providers implement a single-attempt `ProviderBackend`
(`normalized_payload`, `send`). Types are in `apps/core/types.py`.

- **Models by alias only.** `config/models.yaml` is the single place model IDs appear;
  `ModelRegistry.resolve` validates aliases and lists known ones on error.
- **Tool use.** `ToolCall(id, name, arguments)`; a response may carry several, in order.
  Results return as `ToolResult(tool_call_id, content, is_error)`, all results for one
  turn in one user message. The agent loop itself is Phase 2.
- **Opaque blocks.** Current Claude models return thinking blocks that must be echoed
  back unchanged in tool loops. They travel as `OpaqueBlock(provider, data)` and are
  passed through verbatim; sending one to a different provider is an error.
- **Stop reasons** are preserved (`end_turn`, `tool_use`, `max_tokens`, `refusal`, ...);
  unknown values become `unknown` with the raw string kept. Refusal category is kept.
- **Usage is never invented.** A 200 response without usage raises
  `ProviderResponseError`. Failed calls carry no usage. Unreported cache fields stay `None`.
- **Sampling parameters.** Per Anthropic's current docs, Opus 5.5, Sonnet 5.5 and
  Haiku 5.5 reject non-default `temperature`. `temperature` is optional; setting it on a
  model whose config says `supports_temperature: false` raises before any spend. Effort is
  pinned per model in config. The response cache (0003) gives repeatable *replay* of
  recorded outputs; it does not make the model deterministic. Fresh model behavior and
  judge variance still have to be measured with uncached calls.
- **No server-side refusal fallbacks.** A fallback would change which model answered and
  corrupt per-model measurements. Refusals surface as `stop_reason="refusal"`.

## Errors and timeouts
SDK retries are off (`max_retries=0`); `ModelClient` retries so each attempt passes the
budget guard (0004). Per-attempt timeout and retry settings are in `config/models.yaml`.

| Error | When | Retried | Charge |
|---|---|---|---|
| `ProviderTimeoutError` | client timeout, HTTP 504 | yes | uncertain |
| `ProviderConnectionError` | connection failed or dropped | yes | uncertain |
| `ProviderRateLimitError` | 429 (honors `retry-after`) | yes | none |
| `ProviderOverloadedError` | 529 | yes | none |
| `ProviderServerError` | other 5xx | yes | none |
| `ProviderAuthError` | 401, 403 | no | none |
| `ProviderBadRequestError` | other 4xx incl. unknown model | no | none |
| `ProviderResponseError` | 200 but body breaks the contract (missing or invalid usage, unparseable content) | no | uncertain |
| any other exception from a backend | a bug or an interrupt mid-attempt | no | uncertain |

"Charge: none" assumes the API does not bill requests that return an error status. Timeouts,
dropped connections, malformed 200s and unexpected exceptions are treated as possibly
billed. Every failed attempt is recorded as a `FailedAttempt` (type, message, status,
request ID, charge, uncertain cost): on success in `response.execution.failed_attempts`,
on failure on the raised error. `ModelClient` logs each failed attempt through the
standard logging module; entry points install redacting logging (0005).
