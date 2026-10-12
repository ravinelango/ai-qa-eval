"""Response-cache semantics. The disk cache itself is built in Phase 1e.

What is fixed now (docs/decisions/0003-response-cache.md):
- The key covers the provider, the resolved model ID, the complete normalized
  request payload the provider would send, and CACHE_SCHEMA_VERSION.
- Only successful responses are cached. Failed calls are never stored.
- A cached response keeps the original call's GenerationStats (tokens, estimated
  cost, latency); ExecutionStats describes the current execution (cache_hit=True,
  elapsed time of the lookup, zero additional API cost).
- Judge-variance experiments and live latency measurements must run with
  CachePolicy.BYPASS, and the run file records the policy used.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

# Bump when the payload normalization or the stored record format changes, so old
# entries can never be served for a request they no longer describe.
CACHE_SCHEMA_VERSION = 1


class CachePolicy(StrEnum):
    USE = "use"  # read hits, write misses
    BYPASS = "bypass"  # always call the provider; neither read nor write
    REFRESH = "refresh"  # always call the provider; overwrite the stored entry


def cache_key(*, provider: str, resolved_model_id: str, payload: dict[str, Any]) -> str:
    """Deterministic key: identical inputs give the same key regardless of dict order."""
    material = {
        "schema": CACHE_SCHEMA_VERSION,
        "provider": provider,
        "model_id": resolved_model_id,
        "payload": payload,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
