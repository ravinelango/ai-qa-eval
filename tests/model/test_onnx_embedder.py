"""The pinned ONNX embedding model. Opt-in: `uv run pytest tests/model --run-model`.

Needs the pinned files from `make model-fetch`; no network at test time. The
sentences below are written for these tests and are not corpus text.
"""

from __future__ import annotations

import math

import pytest

from apps.rag.config import RagConfig
from apps.rag.embedding import OnnxEmbedder
from apps.rag.errors import EmbedderError


@pytest.fixture(scope="module")
def embedder() -> OnnxEmbedder:
    config = RagConfig.load()
    if not config.embedding.is_pinned:
        pytest.skip("the embedding model is not pinned in config/rag.yaml yet")
    try:
        return OnnxEmbedder(config.embedding, config.root)
    except EmbedderError as exc:
        pytest.skip(f"pinned model files unavailable: {exc}")


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_have_the_configured_size_and_unit_norm(embedder: OnnxEmbedder) -> None:
    [vector] = embedder.embed_documents(["Electronic records need audit trails."])
    assert len(vector) == embedder.settings.dimensions
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-5)


def test_repeated_and_batched_embeddings_agree(embedder: OnnxEmbedder) -> None:
    texts = ["Audit trails record changes.", "Passwords must be unique to one person."]
    once = embedder.embed_documents(texts)
    assert embedder.embed_documents(texts) == once
    alone = embedder.embed_documents(texts[1:])[0]
    assert all(math.isclose(a, b, abs_tol=1e-4) for a, b in zip(alone, once[1], strict=True))


def test_query_is_closer_to_the_related_passage(embedder: OnnxEmbedder) -> None:
    query = embedder.embed_query("Who changed this record and when?")
    related, unrelated = embedder.embed_documents(
        [
            "The system keeps a time-stamped log of each change and the user who made it.",
            "Identification codes and passwords are reviewed periodically.",
        ]
    )
    assert _cosine(query, related) > _cosine(query, unrelated)


def test_counts_include_special_tokens_and_long_input_is_refused(embedder: OnnxEmbedder) -> None:
    assert embedder.count("audit trail") == 4  # [CLS] audit trail [SEP]
    with pytest.raises(EmbedderError, match="would be truncated"):
        embedder.embed_documents(["record " * 600])
