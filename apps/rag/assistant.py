"""The RAG assistant: retrieve, ask the model, parse and check its answer.

`answer(question)` returns an `AnswerRecord` (apps.core.evidence) that can be saved
as-is: the behavior class, the answer text, the cited chunk IDs exactly as the
model gave them, the citations that were not among the retrieved chunks (flagged,
never dropped), the retrieved chunks with text and scores, the prompt version and
SHA-256, the full model call and the trace ID.

Behavior classes: "answer", "dont_know" and "refuse" are the model's own choice.
"invalid" means the output broke the answer contract (not one JSON object of the
required shape, or cut off at max_tokens); the raw text is kept in `answer`, and a
citation list in otherwise malformed JSON is still recorded and checked.
"error" means the model call failed; the error is in `call.error`. An API-level
refusal (stop_reason "refusal") is recorded as "refuse" with `api_refusal=True`.

A provider error ends only this question. Anything else that stops the call (the
budget is exhausted, paid calls are not allowed, an interrupt) stops the run: the
complete record (question, retrieval, prompt identity, resolved call settings and
every failed attempt) is attached to the exception with `stopped_run=True`, where
`answer_record_of` finds it for the run file, and the exception propagates.

A valid citation only shows that the answer points at a chunk that was retrieved.
It does not show that the chunk supports the answer, or that the answer is right.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from apps.core.client import Completer
from apps.core.errors import ProviderError
from apps.core.evidence import (
    AnswerRecord,
    Behavior,
    CallRecord,
    RetrievalRecord,
    call_settings,
    error_record,
)
from apps.core.models import ModelRegistry
from apps.core.tracing.base import TraceAttributes, Tracer
from apps.core.tracing.local import NoopTracer
from apps.core.types import CompletionRequest, CompletionResponse, Message
from apps.rag.config import GenerationSettings
from apps.rag.prompts import PromptTemplate
from apps.rag.retrieve import Retriever

_FENCED = re.compile(r"^```(?:json)?\s*\n(?P<body>.*)\n```$", re.DOTALL)
_RECORD_ATTRIBUTE = "answer_record"


def answer_record_of(error: BaseException) -> AnswerRecord | None:
    """The record of the question that was being answered when `error` stopped the run."""
    record = getattr(error, _RECORD_ATTRIBUTE, None)
    return record if isinstance(record, AnswerRecord) else None


class ModelAnswer(BaseModel):
    """The JSON object the prompt asks the model for."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    behavior: Literal["answer", "dont_know", "refuse"]
    answer: str
    citations: list[str]


class ParsedOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    behavior: Behavior
    answer: str
    citations: list[str]
    error: str | None = None
    api_refusal: bool = False


def parse_model_output(response: CompletionResponse) -> ParsedOutput:
    text = response.text
    if response.stop_reason == "refusal":
        return ParsedOutput(behavior="refuse", answer=text, citations=[], api_refusal=True)
    if response.stop_reason == "max_tokens":
        return ParsedOutput(
            behavior="invalid", answer=text, citations=[], error="output cut off at max_tokens"
        )
    body = text.strip()
    if fenced := _FENCED.match(body):
        body = fenced.group("body").strip()
    try:
        raw = json.loads(body)
    except json.JSONDecodeError as exc:
        return ParsedOutput(behavior="invalid", answer=text, citations=[], error=f"not JSON: {exc}")
    try:
        parsed = ModelAnswer.model_validate(raw)
    except ValidationError as exc:
        # Keep any citation list the output carried, so it can still be flagged.
        cited = raw.get("citations") if isinstance(raw, dict) else None
        citations = [c for c in cited if isinstance(c, str)] if isinstance(cited, list) else []
        return ParsedOutput(
            behavior="invalid",
            answer=text,
            citations=citations,
            error=f"JSON does not match the answer contract: {exc.error_count()} error(s)",
        )
    if parsed.behavior == "answer" and not parsed.answer.strip():
        return ParsedOutput(
            behavior="invalid", answer=text, citations=[], error="behavior answer, empty answer"
        )
    return ParsedOutput(
        behavior=parsed.behavior, answer=parsed.answer, citations=list(parsed.citations)
    )


def unretrieved(citations: list[str], retrieval: RetrievalRecord) -> list[str]:
    retrieved = {r.chunk_id for r in retrieval.results}
    out: list[str] = []
    for citation in citations:
        if citation not in retrieved and citation not in out:
            out.append(citation)
    return out


class RagAssistant:
    def __init__(
        self,
        *,
        retriever: Retriever,
        completer: Completer,
        registry: ModelRegistry,
        prompt: PromptTemplate,
        generation: GenerationSettings,
        tracer: Tracer | None = None,
    ) -> None:
        self.retriever = retriever
        self.completer = completer
        self.registry = registry
        self.prompt = prompt
        self.generation = generation
        self.tracer = tracer or NoopTracer()

    def answer(self, question: str, *, answer_id: str = "q1") -> AnswerRecord:
        with self.tracer.observe(
            "rag.answer",
            input={"question": question},
            metadata={"prompt_sha256": self.prompt.sha256},
            attributes=TraceAttributes(prompt_version=self.prompt.tag),
        ) as root:
            retrieval = self.retriever.retrieve(question)
            request = CompletionRequest(
                model=self.generation.model,
                system=self.prompt.system,
                messages=(
                    Message.user(self.prompt.render_user(question=question, retrieval=retrieval)),
                ),
                max_tokens=self.generation.max_tokens,
            )
            call = CallRecord(
                call_id=f"{answer_id}-generate",
                trace_id=root.trace_id,
                settings=call_settings(self.registry, request),
                request=request,
            )
            try:
                response = self.completer.complete(request)
            except ProviderError as exc:
                call = call.model_copy(update={"error": error_record(exc)})
                parsed = ParsedOutput(
                    behavior="error", answer="", citations=[], error=f"{type(exc).__name__}"
                )
            except BaseException as exc:
                # A stop for the whole run: keep the evidence, then let it propagate.
                call = call.model_copy(update={"error": error_record(exc)})
                parsed = ParsedOutput(
                    behavior="error",
                    answer="",
                    citations=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
                record = self._record(
                    answer_id, question, retrieval, call, parsed, root.trace_id, stopped=True
                )
                setattr(exc, _RECORD_ATTRIBUTE, record)
                root.update(output=_trace_output(record))
                raise
            else:
                call = call.model_copy(update={"response": response})
                parsed = parse_model_output(response)
            record = self._record(answer_id, question, retrieval, call, parsed, root.trace_id)
            root.update(output=_trace_output(record))
        return record

    def _record(
        self,
        answer_id: str,
        question: str,
        retrieval: RetrievalRecord,
        call: CallRecord,
        parsed: ParsedOutput,
        trace_id: str | None,
        *,
        stopped: bool = False,
    ) -> AnswerRecord:
        flagged = unretrieved(parsed.citations, retrieval)
        valid = [c for c in parsed.citations if c not in flagged]
        return AnswerRecord(
            answer_id=answer_id,
            question=question,
            behavior=parsed.behavior,
            answer=parsed.answer,
            cited_chunk_ids=parsed.citations,
            unretrieved_citations=flagged,
            missing_citations=parsed.behavior == "answer" and not valid,
            output_error=parsed.error,
            api_refusal=parsed.api_refusal,
            stopped_run=stopped,
            prompt_version=self.prompt.tag,
            prompt_sha256=self.prompt.sha256,
            retrieval=retrieval,
            call=call,
            trace_id=trace_id,
        )


def _trace_output(record: AnswerRecord) -> dict[str, object]:
    return {
        "behavior": record.behavior,
        "stopped_run": record.stopped_run,
        "cited_chunk_ids": record.cited_chunk_ids,
        "unretrieved_citations": record.unretrieved_citations,
        "retrieved_chunk_ids": [r.chunk_id for r in record.retrieval.results],
    }
