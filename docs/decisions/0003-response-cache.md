# 0003 — Response cache semantics

**Status:** semantics accepted in Phase 0; disk cache implemented in Phase 1e.

## Decision
- **Key** (`apps/core/cache.py::cache_key`): SHA-256 over canonical JSON of the provider
  name, the resolved model ID, the complete normalized request payload the backend would
  send (`ProviderBackend.normalized_payload`), and `CACHE_SCHEMA_VERSION`. Any change to
  model, prompt text, tools, max_tokens or effort changes the key.
- **Only successful responses are cached.** Errors, refusals-by-error and partial results
  are never stored. (A `stop_reason="refusal"` response is a successful API result and is
  cached like any other.)
- **Two sets of measurements.** `CompletionResponse.generation` (`GenerationStats`) always
  describes the original API call: tokens, estimated cost, latency, pricing date.
  `CompletionResponse.execution` (`ExecutionStats`) describes this execution: `cache_hit`,
  elapsed time, attempts, additional estimated API cost (0 on a hit), uncertain cost.
  Reports must use `generation` for quality/cost-per-request comparisons and must not
  present cached elapsed times as model latency.
- **Explicit bypass.** `CachePolicy` is `use`, `bypass` or `refresh`. Judge-variance
  experiments and live latency measurements must run with `bypass`; the policy is recorded
  in every run file manifest.
