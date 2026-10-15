"""Cross-encoder reranking, for retrieval.mode hybrid_rerank.

`OnnxReranker` runs the pinned cross-encoder (config/rag.yaml, reranker) through
onnxruntime. It reads a question and a passage together and returns one relevance
score per passage: the model's logit, higher meaning more relevant. Like the
embedder, it loads only files that match their pins, and the default test suite
never loads it (FORBID_MODEL_LOAD_ENV).

Each pair must fit the model's input limit. When one does not, the end of the
passage is cut (never the question) and the number of tokens cut is returned with
the score, so the retrieval record shows it. A question too long to leave room for
any passage is refused.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from apps.rag.config import RerankerSettings
from apps.rag.embedding import load_onnx_session, load_tokenizer, pinned_path
from apps.rag.errors import EmbedderError


@dataclass(frozen=True)
class RerankScore:
    score: float
    # Passage tokens cut so the pair fits the model's input limit.
    truncated_tokens: int


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def fingerprint(self) -> str:
        """Identifies everything that determines the scores."""
        ...

    def score(self, query: str, passages: Sequence[str]) -> list[RerankScore]: ...


class OnnxReranker:
    def __init__(self, settings: RerankerSettings, root: Path) -> None:
        self.settings = settings
        tokenizer_path = pinned_path(settings, root, "tokenizer")
        # One tokenizer counts the whole pair; the other cuts the passage to fit.
        self._counting = load_tokenizer(tokenizer_path)
        self._cutting = load_tokenizer(tokenizer_path)
        self._cutting.enable_truncation(settings.max_tokens, strategy="only_second")
        pad_id = self._counting.token_to_id("[PAD]")
        if pad_id is None:
            raise EmbedderError("the pinned reranker tokenizer has no [PAD] token")
        self._pad_id = pad_id
        self._session = load_onnx_session(pinned_path(settings, root, "model"), settings.threads)

    @property
    def name(self) -> str:
        return self.settings.name

    @property
    def fingerprint(self) -> str:
        return self.settings.fingerprint()

    def score(self, query: str, passages: Sequence[str]) -> list[RerankScore]:
        # [CLS] question [SEP] passage [SEP]: the passage needs at least one token.
        question = len(self._counting.encode(query, add_special_tokens=True).ids)
        if question + 2 > self.settings.max_tokens:
            raise EmbedderError(
                f"the question has {question} tokens, too many to pair with any passage "
                f"within the reranker's {self.settings.max_tokens}-token limit"
            )
        scores: list[RerankScore] = []
        size = self.settings.batch_size
        for start in range(0, len(passages), size):
            scores += self._score_batch(query, passages[start : start + size])
        return scores

    def _score_batch(self, query: str, passages: Sequence[str]) -> list[RerankScore]:
        import numpy as np

        pairs = [(query, passage) for passage in passages]
        full = self._counting.encode_batch(pairs, add_special_tokens=True)
        encodings = self._cutting.encode_batch(pairs, add_special_tokens=True)
        width = max(len(e.ids) for e in encodings)
        ids = np.full((len(encodings), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(encodings), width), dtype=np.int64)
        types = np.zeros((len(encodings), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            n = len(encoding.ids)
            ids[row, :n] = encoding.ids
            mask[row, :n] = encoding.attention_mask
            types[row, :n] = encoding.type_ids
        (logits,) = self._session.run(
            ["logits"], {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        )
        if logits.shape != (len(encodings), 1):
            raise EmbedderError(f"unexpected reranker output shape {logits.shape}")
        return [
            RerankScore(float(logit), len(f.ids) - len(e.ids))
            for logit, f, e in zip(logits[:, 0], full, encodings, strict=True)
        ]
