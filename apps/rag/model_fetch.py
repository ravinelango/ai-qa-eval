"""Explicit download of the pinned embedding model files.

    make model-fetch                                   (download and verify pins)
    uv run python -m apps.rag.model_fetch --pin        (first time only)

Files go to `embedding.local_dir` (gitignored). Each is fetched from the exact
repository revision in config/rag.yaml and must match its SHA-256 pin, or it is
not installed.

`--pin` is for choosing the pins: it looks up the repository's current revision,
downloads the files at that revision into a temporary directory, and prints the
revision and SHA-256 values to record in config/rag.yaml. It changes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from apps.core.redact import configure_logging
from apps.rag.config import EmbeddingSettings, PinnedFile, RagConfig
from apps.rag.download import Fetcher, https_fetch
from apps.rag.errors import EmbedderError, RagError

HF = "https://huggingface.co"


def resolve_url(repo: str, revision: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/{revision}/{path}"


def current_revision(repo: str, fetch: Fetcher) -> str:
    data = json.loads(fetch(f"{HF}/api/models/{repo}"))
    sha = data.get("sha")
    if not isinstance(sha, str) or len(sha) != 40:
        raise EmbedderError(f"could not read the current revision of {repo}")
    return sha


@dataclass(frozen=True)
class FetchedFile:
    path: str
    sha256: str
    bytes: int


def download_files(
    repo: str, revision: str, files: list[tuple[PinnedFile, Path]], fetch: Fetcher
) -> list[FetchedFile]:
    """Download each file at `revision` to its target path; install only those matching
    a set pin."""
    fetched: list[FetchedFile] = []
    for pin, target in files:
        data = fetch(resolve_url(repo, revision, pin.path))
        digest = hashlib.sha256(data).hexdigest()
        if pin.sha256 is not None and digest != pin.sha256:
            raise EmbedderError(
                f"{repo}@{revision}/{pin.path}: downloaded SHA-256 {digest} does not match "
                f"the pin {pin.sha256}; nothing was installed for this file"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        fetched.append(FetchedFile(pin.path, digest, len(data)))
    return fetched


def fetch_pinned(settings: EmbeddingSettings, root: Path, fetch: Fetcher) -> list[FetchedFile]:
    if not settings.is_pinned or settings.revision is None:
        raise EmbedderError(
            "embedding.revision and both SHA-256 pins must be set in config/rag.yaml first; "
            "run with --pin to look them up"
        )
    # The tokenizer first: it alone is enough to build the chunk manifest.
    pins = [settings.tokenizer, settings.model]
    return download_files(
        settings.repo, settings.revision, [(p, settings.file_path(p, root)) for p in pins], fetch
    )


def propose_pins(settings: EmbeddingSettings, fetch: Fetcher) -> tuple[str, list[FetchedFile]]:
    revision = current_revision(settings.repo, fetch)
    unpinned = [PinnedFile(path=p.path) for p in (settings.model, settings.tokenizer)]
    with tempfile.TemporaryDirectory() as tmp:
        targets = [(p, Path(tmp) / p.path) for p in unpinned]
        files = download_files(settings.repo, revision, targets, fetch)
    return revision, files


def main(argv: list[str] | None = None, *, fetch: Fetcher = https_fetch) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--pin", action="store_true", help="look up pins; change nothing")
    args = parser.parse_args(argv)
    config = RagConfig.load()
    settings = config.embedding
    try:
        if args.pin:
            revision, files = propose_pins(settings, fetch)
            print(f"Record in config/rag.yaml (embedding section) for {settings.repo}:")
            print(f"  revision: {revision}")
            for f in files:
                print(f"  {f.path}: sha256 {f.sha256} ({f.bytes} bytes)")
            return 0
        for f in fetch_pinned(settings, config.root, fetch):
            print(f"installed {f.path} sha256={f.sha256} ({f.bytes} bytes)")
    except (RagError, OSError) as exc:  # OSError covers network failures (URLError)
        print(f"model fetch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
