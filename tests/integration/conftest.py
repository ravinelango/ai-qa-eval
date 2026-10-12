"""Offline integration tests must not depend on the host's proxy configuration.

Every test here runs with a hostile SOCKS proxy in the environment. A client that
inherits proxy settings from the environment fails fast (no SOCKS support is
installed), so any test that silently relied on env proxies would break here
rather than on someone else's machine. Socket blocking stays on as well.
"""

from __future__ import annotations

import pytest

HOSTILE_PROXY = "socks5://127.0.0.1:9"
PROXY_VARS = ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")


@pytest.fixture(autouse=True)
def hostile_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_VARS:
        monkeypatch.setenv(name, HOSTILE_PROXY)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
