"""HTTPS downloads for the two explicit refresh commands (corpus, local models).

Nothing else in the pipeline touches the network. Callers take a `Fetcher` so
tests can supply responses without any network access.
"""

from __future__ import annotations

import gzip
import http.client
import urllib.request
from collections.abc import Callable

from apps.rag.errors import RagError

# The "Mozilla/5.0 (compatible; ...)" form is the usual crawler format; fda.gov refuses
# user agents without it. It still names this tool and links the repo.
USER_AGENT = "Mozilla/5.0 (compatible; ai-qa-eval/0.1; +https://github.com/ravinelango/ai-qa-eval)"

Fetcher = Callable[[str], bytes]


def https_fetch(url: str, *, timeout_s: float = 120.0) -> bytes:
    """GET `url` (HTTPS only; redirects are followed) and return the decoded body.

    Compression is requested because some endpoints (the eCFR API) refuse
    uncompressed responses.
    """
    if not url.startswith("https://"):
        raise RagError(f"refusing to fetch a non-HTTPS URL: {url}")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    request = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        try:
            body: bytes = response.read()
        except http.client.IncompleteRead as exc:
            raise RagError(
                f"the download of {url} was cut off after {len(exc.partial)} bytes; "
                "nothing was installed, try again"
            ) from exc
        encoding = response.headers.get("Content-Encoding")
    return decode_body(body, encoding)


def decode_body(body: bytes, content_encoding: str | None) -> bytes:
    encoding = (content_encoding or "identity").strip().lower()
    if encoding == "gzip":
        return gzip.decompress(body)
    if encoding == "identity":
        return body
    raise RagError(f"unsupported Content-Encoding {content_encoding!r}")
