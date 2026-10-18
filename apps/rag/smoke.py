"""Retrieval smoke check: does the real index return the expected sections?

    make retrieval-smoke

Runs every query in evals/dev_checks/retrieval_smoke.yaml (or --spec) against the
local index with the pinned local models (no API calls, no network) and writes a
report with
the full retrieval for each query: chunk IDs, labels, text and scores. The queries
are developer-authored and not human-reviewed; this is a plumbing check, not an
evaluation. top_k and the thresholds come from the spec file, which is fixed before
the check is run.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from apps.core.cache import CachePolicy
from apps.core.errors import ConfigurationError
from apps.core.evidence import RetrievalRecord, RunManifest, git_state, new_run_id
from apps.core.models import REPO_ROOT, ModelRegistry
from apps.core.redact import configure_logging, redact
from apps.rag.config import RagConfig
from apps.rag.embedding import OnnxEmbedder
from apps.rag.errors import RagError
from apps.rag.rerank import OnnxReranker
from apps.rag.retrieve import Retriever, build_retriever

DEFAULT_SPEC = REPO_ROOT / "evals" / "dev_checks" / "retrieval_smoke.yaml"
# 2: retrieval records carry the ranking mode, fingerprint and per-chunk details.
# 3: retrieval records carry the reranker and each chunk's fused and reranker scores.
# 4: retrieval records carry the expanded keyword query (hybrid_prf).
REPORT_SCHEMA_VERSION = 4

QueryKind = Literal["section_lookup", "paraphrase"]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Expectation(_Strict):
    doc_id: str
    locator: str


class SmokeQuery(_Strict):
    id: str = Field(pattern=r"^[A-Z]+\d+$")
    kind: QueryKind
    query: str = Field(min_length=1)
    expect: tuple[Expectation, ...] = Field(min_length=1)


class SmokeSpec(_Strict):
    purpose: str
    authored_by: str
    # True only when the owner has reviewed and approved the queries and expectations.
    reviewed_by_human: bool
    top_k: int = Field(ge=1, le=50)
    min_hit_rate: dict[QueryKind, float]
    queries: tuple[SmokeQuery, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> SmokeSpec:
        ids = [q.id for q in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("query ids must be unique")
        kinds = {q.kind for q in self.queries}
        if kinds != set(self.min_hit_rate):
            raise ValueError("every query kind needs a threshold, and every threshold queries")
        if not all(0.0 < rate <= 1.0 for rate in self.min_hit_rate.values()):
            raise ValueError("thresholds must be in (0, 1]")
        return self


def load_spec(path: Path) -> tuple[SmokeSpec, str]:
    try:
        data = path.read_bytes()
        return SmokeSpec.model_validate(yaml.safe_load(data)), hashlib.sha256(data).hexdigest()
    except (OSError, ValidationError) as exc:
        raise ConfigurationError(f"invalid smoke spec {path}: {exc}") from exc


def matches(doc_id: str, locator: str, expectation: Expectation) -> bool:
    """Same document, and the locator equals the expected one or is nested under it:
    a paragraph "(", a guidance subsection ".", a split part "#", or the section's
    introductory text "-intro". Slug locators such as "scope-of-application" do not
    nest under "scope"."""
    if doc_id != expectation.doc_id:
        return False
    want = expectation.locator
    if locator in (want, f"{want}-intro") or locator.startswith(f"{want}-intro#"):
        return True
    return any(locator.startswith(want + sep) for sep in "(.#")


class SmokeResult(_Strict):
    query_id: str
    kind: QueryKind
    query: str
    expect: tuple[Expectation, ...]
    hit: bool
    # Rank of the first matching chunk, if any.
    hit_rank: int | None
    retrieval: RetrievalRecord


class KindSummary(_Strict):
    kind: QueryKind
    queries: int
    hits: int
    hit_rate: float
    min_hit_rate: float
    passed: bool


class SmokeReport(_Strict):
    schema_version: Literal[4] = 4
    manifest: RunManifest
    spec_path: str
    spec_sha256: str
    authored_by: str
    reviewed_by_human: bool
    top_k: int
    results: tuple[SmokeResult, ...]
    summary: tuple[KindSummary, ...]
    passed: bool


def run_queries(spec: SmokeSpec, retriever: Retriever) -> list[SmokeResult]:
    results: list[SmokeResult] = []
    for query in spec.queries:
        retrieval = retriever.retrieve(query.query, top_k=spec.top_k)
        rank: int | None = None
        for r in retrieval.results:
            locator = retriever.chunk(r.chunk_id).locator
            if any(matches(r.doc_id, locator, e) for e in query.expect):
                rank = r.rank
                break
        results.append(
            SmokeResult(
                query_id=query.id,
                kind=query.kind,
                query=query.query,
                expect=query.expect,
                hit=rank is not None,
                hit_rank=rank,
                retrieval=retrieval,
            )
        )
    return results


def summarize(spec: SmokeSpec, results: list[SmokeResult]) -> tuple[KindSummary, ...]:
    summaries: list[KindSummary] = []
    for kind, threshold in sorted(spec.min_hit_rate.items()):
        of_kind = [r for r in results if r.kind == kind]
        hits = sum(r.hit for r in of_kind)
        rate = hits / len(of_kind)
        summaries.append(
            KindSummary(
                kind=kind,
                queries=len(of_kind),
                hits=hits,
                hit_rate=rate,
                min_hit_rate=threshold,
                passed=rate >= threshold,
            )
        )
    return tuple(summaries)


def build_report(
    spec: SmokeSpec,
    spec_path: Path,
    spec_sha256: str,
    retriever: Retriever,
    *,
    synthetic: bool,
    notes: str = "",
) -> SmokeReport:
    results = run_queries(spec, retriever)
    summary = summarize(spec, results)
    registry = ModelRegistry.load()
    sha, dirty = git_state()
    manifest = RunManifest(
        run_id=new_run_id(),
        kind="retrieval_smoke",
        synthetic=synthetic,
        created_at=datetime.now(UTC),
        git_sha=sha,
        git_dirty=dirty,
        models_config_sha256=registry.source_sha256,
        pricing_checked=registry.pricing_checked,
        tracing_backend=retriever.tracer.backend,
        cache_policy=CachePolicy.BYPASS,
        corpus_fingerprint=retriever.manifest.corpus_fingerprint,
        chunk_manifest_sha256=retriever.manifest_sha256,
        embedding_fingerprint=retriever.embedder.fingerprint,
        retrieval_fingerprint=retriever.fingerprint,
        notes=notes,
    )
    try:
        shown_path = str(spec_path.relative_to(REPO_ROOT))
    except ValueError:
        shown_path = str(spec_path)
    return SmokeReport(
        manifest=manifest,
        spec_path=shown_path,
        spec_sha256=spec_sha256,
        authored_by=spec.authored_by,
        reviewed_by_human=spec.reviewed_by_human,
        top_k=spec.top_k,
        results=tuple(results),
        summary=summary,
        passed=all(s.passed for s in summary),
    )


def write_report(report: SmokeReport, out_dir: Path) -> Path:
    path = out_dir / f"{report.manifest.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact(report.model_dump_json(indent=2)) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "retrieval_smoke")
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "hybrid_rerank", "hybrid_prf"],
        help="retrieval mode for this run (default: retrieval.mode in config/rag.yaml)",
    )
    args = parser.parse_args(argv)
    config = RagConfig.load()
    if args.mode is not None:
        retrieval = config.retrieval.model_copy(update={"mode": args.mode})
        config = config.model_copy(update={"retrieval": retrieval})
    try:
        spec, spec_sha = load_spec(args.spec)
        reranker = (
            OnnxReranker(config.reranker, config.root)
            if config.retrieval.mode == "hybrid_rerank"
            else None
        )
        retriever = build_retriever(
            config,
            OnnxEmbedder(config.embedding, config.root),
            tokenizer_fingerprint=config.embedding.tokenizer_fingerprint(),
            reranker=reranker,
        )
        report = build_report(
            spec,
            args.spec,
            spec_sha,
            retriever,
            synthetic=False,
            notes=(
                "Developer-authored retrieval check; not an evaluation result. "
                f"Retrieval mode: {config.retrieval.mode}."
            ),
        )
    except (RagError, ConfigurationError) as exc:
        print(f"retrieval smoke check could not run: {exc}", file=sys.stderr)
        return 2
    path = write_report(report, args.out)
    for r in report.results:
        top = ", ".join(f"{x.chunk_id} ({x.score:.3f})" for x in r.retrieval.results)
        mark = f"hit@{r.hit_rank}" if r.hit else "MISS"
        print(f"{r.query_id} {mark:7} {r.query}\n          top-{report.top_k}: {top}")
    for s in report.summary:
        verdict = "pass" if s.passed else "FAIL"
        print(
            f"{s.kind}: {s.hits}/{s.queries} = {s.hit_rate:.2f} "
            f"(min {s.min_hit_rate:.2f}) {verdict}"
        )
    print(f"overall: {'PASS' if report.passed else 'FAIL'}; report: {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
