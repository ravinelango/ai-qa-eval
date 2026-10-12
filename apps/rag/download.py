"""HTTPS downloads for the two explicit refresh commands (corpus, embedding model).

Nothing else in the pipeline touches the network. Callers take a `Fetcher` so
tests can supply responses without any network access.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable

from apps.rag.errors import RagError

USER_AGENT = "ai-qa-eval (https://github.com/ravinelango/ai-qa-eval)"

Fetcher = Callable[[str], bytes]


def https_fetch(url: str, *, timeout_s: float = 120.0) -> bytes:
    """GET `url` (HTTPS only; redirects are followed) and return the body."""
    if not url.startswith("https://"):
        raise RagError(f"refusing to fetch a non-HTTPS URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        body: bytes = response.read()
    return body
