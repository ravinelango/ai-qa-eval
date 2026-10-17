"""The pinned ONNX cross-encoder. Opt-in: `uv run pytest tests/model --run-model`.

Needs the pinned files from `make model-fetch`; no network at test time. The
sentences below are written for these tests and are not corpus text.
"""

from __future__ import annotations

import math

import pytest

from apps.rag.config import RagConfig
from apps.rag.errors import EmbedderError
from apps.rag.rerank import OnnxReranker

QUESTION = "Who should review the audit trail of changes to laboratory data?"
RELATED = "Quality unit staff should review audit trails that record changes to lab data."
UNRELATED = "The cafeteria menu changes every Tuesday and lists vegetarian options."


@pytest.fixture(scope="module")
def reranker() -> OnnxReranker:
    config = RagConfig.load()
    try:
        return OnnxReranker(config.reranker, config.root)
    except EmbedderError as exc:
        pytest.skip(f"pinned reranker files unavailable: {exc}")


def test_the_related_passage_scores_higher(reranker: OnnxReranker) -> None:
    related, unrelated = reranker.score(QUESTION, [RELATED, UNRELATED])
    assert related.score > unrelated.score
    assert (related.truncated_tokens, unrelated.truncated_tokens) == (0, 0)


def test_repeated_and_batched_scores_agree(reranker: OnnxReranker) -> None:
    once = reranker.score(QUESTION, [RELATED, UNRELATED])
    assert reranker.score(QUESTION, [RELATED, UNRELATED]) == once
    alone = reranker.score(QUESTION, [UNRELATED])[0]
    assert math.isclose(alone.score, once[1].score, abs_tol=1e-4)


def test_a_long_passage_is_cut_and_the_cut_is_reported(reranker: OnnxReranker) -> None:
    [result] = reranker.score(QUESTION, ["audit " * 600])
    # [CLS] + 12 question tokens + [SEP], 600 passage tokens, [SEP]: 615 for 512.
    assert result.truncated_tokens == 615 - 512
    assert math.isfinite(result.score)


def test_a_question_with_no_room_for_a_passage_is_refused(reranker: OnnxReranker) -> None:
    with pytest.raises(EmbedderError, match="too many to pair"):
        reranker.score("audit " * 520, [RELATED])
