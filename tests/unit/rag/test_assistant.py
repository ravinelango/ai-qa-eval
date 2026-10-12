"""The RAG assistant with the scripted fake provider and an in-memory index.

Model replies here are scripted strings, so these tests check plumbing: parsing,
citation flags, evidence and tracing. They say nothing about answer quality.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from apps.core.cache import CachePolicy
from apps.core.client import ModelClient
from apps.core.errors import PaidCallsNotAllowedError, ProviderBadRequestError
from apps.core.evidence import RunFile, RunManifest, read_run_file, write_run_file
from apps.core.models import ModelRegistry
from apps.core.providers.fake import FakeBackend, ScriptedReply, Step
from apps.core.tracing.local import RecordingTracer
from apps.core.types import TextBlock
from apps.rag.assistant import RagAssistant
from apps.rag.chunking import build_manifest, chunk_corpus
from apps.rag.config import GenerationSettings
from apps.rag.corpus import SourceDocument
from apps.rag.embedding import HashingEmbedder
from apps.rag.index import IndexStamp
from apps.rag.parse_ecfr import parse_ecfr_part
from apps.rag.prompts import PROMPTS_DIR, load_prompt
from apps.rag.retrieve import Retriever
from tests.rag_helpers import ECFR_DOC, SYNTHETIC_ECFR, InMemoryIndex, frozen_entry

EMBEDDER = HashingEmbedder(max_tokens=128)
PROMPT = load_prompt(PROMPTS_DIR / "answer-v1.yaml")
QUESTION = "What must the placeholder change logs record?"
EXPECTED_TOP = "synthetic-part99/99.10(e)"


def _retriever(tracer: RecordingTracer | None = None) -> Retriever:
    document = SourceDocument.model_validate(frozen_entry(ECFR_DOC, SYNTHETIC_ECFR.read_bytes()))
    units = parse_ecfr_part(SYNTHETIC_ECFR.read_bytes(), doc_id=document.doc_id, part="99")
    chunks = chunk_corpus([(document, units)], EMBEDDER)
    manifest = build_manifest(
        chunks,
        chunker_version=1,
        corpus_fingerprint="c" * 64,
        tokenizer_fingerprint=EMBEDDER.fingerprint,
        max_tokens=EMBEDDER.max_tokens,
    )
    stamp = IndexStamp(EMBEDDER.name, EMBEDDER.fingerprint, "m" * 64, "c" * 64)
    index = InMemoryIndex(stamp)
    index.add(chunks, EMBEDDER.embed_documents([c.embed_text for c in chunks]))
    return Retriever(
        embedder=EMBEDDER,
        index=index,
        manifest=manifest,
        manifest_sha256="m" * 64,
        top_k=3,
        tracer=tracer,
    )


def _assistant(
    script: list[Step], tracer: RecordingTracer | None = None, model: str = "fake"
) -> tuple[RagAssistant, FakeBackend]:
    registry = ModelRegistry.load()
    backend = FakeBackend(script)
    client = ModelClient(
        registry=registry, backends={"fake": backend}, tracer=tracer, sleep=lambda _: None
    )
    assistant = RagAssistant(
        retriever=_retriever(tracer),
        completer=client,
        registry=registry,
        prompt=PROMPT,
        generation=GenerationSettings(model=model, max_tokens=512, prompt=Path("unused")),
        tracer=tracer,
    )
    return assistant, backend


def _reply(behavior: str, answer: str, citations: list[str]) -> ScriptedReply:
    return ScriptedReply.text(
        json.dumps({"behavior": behavior, "answer": answer, "citations": citations})
    )


def test_answer_with_valid_citations() -> None:
    assistant, backend = _assistant([_reply("answer", "[SCRIPTED] logs.", [EXPECTED_TOP])])
    record = assistant.answer(QUESTION)
    assert record.behavior == "answer"
    assert record.cited_chunk_ids == [EXPECTED_TOP]
    assert record.unretrieved_citations == []
    assert record.missing_citations is False
    assert record.retrieval.results[0].chunk_id == EXPECTED_TOP
    assert (record.prompt_version, record.prompt_sha256) == ("answer@v1", PROMPT.sha256)
    # The request carries the versioned prompt and the retrieved excerpts.
    [request] = backend.requests
    assert request.system == PROMPT.system
    user = request.messages[0].content[0]
    assert isinstance(user, TextBlock)
    assert f'<excerpt id="{EXPECTED_TOP}" kind="regulation" label="§99.10(e)">' in user.text
    assert user.text.endswith(f"Question: {QUESTION}")


def test_citations_outside_the_retrieved_chunks_are_flagged_not_dropped() -> None:
    cited = [EXPECTED_TOP, "synthetic-part99/99.300(a)", "made-up/id", "made-up/id"]
    assistant, _ = _assistant([_reply("answer", "[SCRIPTED]", cited)])
    record = assistant.answer(QUESTION)
    assert record.cited_chunk_ids == cited
    assert record.unretrieved_citations == ["synthetic-part99/99.300(a)", "made-up/id"]
    assert record.missing_citations is False

    assistant, _ = _assistant([_reply("answer", "[SCRIPTED]", ["made-up/id"])])
    assert assistant.answer(QUESTION).missing_citations is True
    assistant, _ = _assistant([_reply("answer", "[SCRIPTED]", [])])
    assert assistant.answer(QUESTION).missing_citations is True


@pytest.mark.parametrize("behavior", ["dont_know", "refuse"])
def test_dont_know_and_refuse(behavior: str) -> None:
    assistant, _ = _assistant([_reply(behavior, "[SCRIPTED] reason.", [])])
    record = assistant.answer(QUESTION)
    assert record.behavior == behavior
    assert record.missing_citations is False


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (ScriptedReply.text("The answer is yes."), "not JSON"),
        (ScriptedReply.text('{"behavior": "answer", "answer": "x"}'), "answer contract"),
        (ScriptedReply.text('{"behavior": "maybe", "answer": "", "citations": []}'), "contract"),
        (
            ScriptedReply.text('{"behavior": "answer", "answer": "x", "citations": [], "n": 1}'),
            "contract",
        ),
        (ScriptedReply.text('{"behavior": "answer", "answer": " ", "citations": []}'), "empty"),
        (
            ScriptedReply(content=_reply("answer", "x", []).content, stop_reason="max_tokens"),
            "max_tokens",
        ),
    ],
)
def test_output_that_breaks_the_contract_is_invalid_and_kept(
    reply: ScriptedReply, error: str
) -> None:
    assistant, _ = _assistant([reply])
    record = assistant.answer(QUESTION)
    assert record.behavior == "invalid"
    assert record.output_error is not None
    assert error in record.output_error
    assert record.call.response is not None
    assert record.answer == record.call.response.text  # raw text kept verbatim


def test_fenced_json_is_accepted() -> None:
    body = json.dumps({"behavior": "answer", "answer": "[SCRIPTED]", "citations": [EXPECTED_TOP]})
    assistant, _ = _assistant([ScriptedReply.text(f"```json\n{body}\n```")])
    assert assistant.answer(QUESTION).behavior == "answer"


def test_api_refusal_and_provider_errors_are_recorded() -> None:
    refusal = ScriptedReply(content=(), stop_reason="refusal", refusal_category="test")
    assistant, _ = _assistant([refusal])
    record = assistant.answer(QUESTION)
    assert (record.behavior, record.api_refusal) == ("refuse", True)

    failure = ProviderBadRequestError("HTTP 400 (scripted)", provider="fake", status_code=400)
    assistant, _ = _assistant([failure])
    record = assistant.answer(QUESTION)
    assert record.behavior == "error"
    assert record.call.response is None
    assert record.call.error is not None
    assert record.call.error["error_type"] == "ProviderBadRequestError"
    assert len(record.call.error["failed_attempts"]) == 1
    assert len(record.retrieval.results) == 3  # retrieval evidence survives the failure


def test_paid_generator_is_refused_without_explicit_opt_in() -> None:
    assistant, backend = _assistant([], model="sonnet")
    with pytest.raises(PaidCallsNotAllowedError):
        assistant.answer(QUESTION)
    assert backend.requests == []


def test_trace_records_prompt_version_and_retrieved_chunk_ids() -> None:
    tracer = RecordingTracer()
    assistant, _ = _assistant([_reply("answer", "[SCRIPTED]", [EXPECTED_TOP])], tracer)
    record = assistant.answer(QUESTION)
    by_name = {r.name: r for r in tracer.records}
    assert set(by_name) == {"rag.answer", "rag.retrieve", "llm.complete"}
    assert {r.trace_id for r in tracer.records} == {record.trace_id}
    root = by_name["rag.answer"]
    assert root.attributes is not None
    assert root.attributes["prompt_version"] == "answer@v1"
    assert root.metadata["prompt_sha256"] == PROMPT.sha256
    retrieved = by_name["rag.retrieve"]
    assert retrieved.kind == "retriever"
    assert [o["chunk_id"] for o in retrieved.output] == [
        r.chunk_id for r in record.retrieval.results
    ]
    assert root.output["retrieved_chunk_ids"] == [r.chunk_id for r in record.retrieval.results]


def test_run_file_stands_alone_with_fingerprints_and_retrieved_text(tmp_path: Path) -> None:
    assistant, _ = _assistant([_reply("answer", "[SCRIPTED]", [EXPECTED_TOP])])
    record = assistant.answer(QUESTION)
    registry = ModelRegistry.load()
    run = RunFile(
        manifest=RunManifest(
            run_id="test-run",
            kind="rag_answer_test",
            synthetic=True,
            created_at=datetime(2026, 1, 5, tzinfo=UTC),
            git_sha=None,
            git_dirty=None,
            models_config_sha256=registry.source_sha256,
            pricing_checked=registry.pricing_checked,
            tracing_backend="none",
            cache_policy=CachePolicy.BYPASS,
            prompt_version=PROMPT.tag,
            prompt_sha256=PROMPT.sha256,
            corpus_fingerprint="c" * 64,
            chunk_manifest_sha256="m" * 64,
            embedding_fingerprint=EMBEDDER.fingerprint,
        ),
        calls=[record.call],
        answers=[record],
    )
    path = tmp_path / "run.json"
    write_run_file(path, run)
    saved = read_run_file(path)
    assert saved == run
    raw = json.loads(path.read_text())
    assert raw["schema_version"] == 3
    top = raw["answers"][0]["retrieval"]["results"][0]
    assert top["chunk_id"] == EXPECTED_TOP
    assert top["text"].startswith("(e) Use of placeholder change logs")
    assert isinstance(top["score"], float)
    assert raw["manifest"]["prompt_sha256"] == PROMPT.sha256
    assert raw["manifest"]["embedding_fingerprint"] == EMBEDDER.fingerprint
