"""The retrieval smoke check end to end on the synthetic corpus (real Chroma, hashing
test embedder). Only `make retrieval-smoke` on the real index produces a real result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apps.rag.embedding import HashingEmbedder
from apps.rag.ingest import ingest
from apps.rag.retrieve import build_retriever
from apps.rag.smoke import SmokeReport, SmokeSpec, build_report, write_report
from tests.rag_helpers import synthetic_corpus

SYNTHETIC_SPEC: dict[str, Any] = {
    "purpose": "test",
    "authored_by": "test",
    "reviewed_by_human": False,
    "top_k": 2,
    "min_hit_rate": {"section_lookup": 1.0, "paraphrase": 0.5},
    "queries": [
        {
            "id": "L1",
            "kind": "section_lookup",
            "query": "placeholder change logs earlier value",
            "expect": [{"doc_id": "synthetic-part99", "locator": "99.10(e)"}],
        },
        {
            "id": "P1",
            "kind": "paraphrase",
            "query": "placeholder token testing rule",
            "expect": [{"doc_id": "synthetic-part99", "locator": "99.300"}],
        },
        {
            "id": "P2",
            "kind": "paraphrase",
            "query": "placeholder token testing rule",
            "expect": [{"doc_id": "synthetic-guidance", "locator": "I"}],
        },
    ],
}


def test_report_on_the_synthetic_corpus(tmp_path: Path) -> None:
    config = synthetic_corpus(tmp_path)
    embedder = HashingEmbedder(max_tokens=64)
    ingest(config, embedder, tokenizer_fingerprint=embedder.fingerprint)
    spec = SmokeSpec.model_validate(SYNTHETIC_SPEC)
    report = build_report(
        spec,
        tmp_path / "spec.yaml",
        "s" * 64,
        build_retriever(config, embedder, tokenizer_fingerprint=embedder.fingerprint),
        synthetic=True,
    )
    hits = {r.query_id: (r.hit, r.hit_rank) for r in report.results}
    assert hits["L1"] == (True, 1)
    assert hits["P1"][0] is True
    assert hits["P2"] == (False, None)
    summary = {s.kind: (s.hits, s.queries, s.passed) for s in report.summary}
    assert summary == {"paraphrase": (1, 2, True), "section_lookup": (1, 1, True)}
    assert report.passed is True
    assert report.manifest.synthetic is True
    assert report.manifest.embedding_fingerprint == embedder.fingerprint
    assert report.manifest.retrieval_fingerprint == config.retrieval.fingerprint()
    assert all(r.retrieval.mode == "hybrid" for r in report.results)
    assert all(len(r.retrieval.results) == 2 for r in report.results)

    path = write_report(report, tmp_path / "out")
    assert SmokeReport.model_validate_json(path.read_text()) == report

    stricter = SmokeSpec.model_validate(
        {**SYNTHETIC_SPEC, "min_hit_rate": {"section_lookup": 1.0, "paraphrase": 0.6}}
    )
    failing = build_report(
        stricter,
        tmp_path / "s.yaml",
        "s" * 64,
        build_retriever(config, embedder, tokenizer_fingerprint=embedder.fingerprint),
        synthetic=True,
    )
    assert failing.passed is False
