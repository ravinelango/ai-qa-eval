"""Embedders: the pinned local ONNX model, and a hashing test double.

Both refuse input over their token limit instead of truncating it, and both count
tokens the way they embed, so the chunker can size chunks exactly.

`OnnxEmbedder` runs bge-small-en-v1.5 through onnxruntime (no PyTorch). It loads
only files whose repository revision and SHA-256 are pinned in config/rag.yaml, and
refuses to start while any pin is missing or a file does not match. The default
test suite never loads it: tests/conftest.py sets FORBID_MODEL_LOAD_ENV.

`HashingEmbedder` is for tests and offline plumbing only. Its vectors encode which
words two texts share; they carry no meaning and its results are never evidence
about retrieval quality.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from apps.rag.config import EmbeddingSettings, canonical_sha256, file_sha256
from apps.rag.errors import EmbedderError

if TYPE_CHECKING:
    from tokenizers import Tokenizer


# The default test suite sets this to "1" (tests/conftest.py), so no test can load the
# real model, however it reaches the loader.
FORBID_MODEL_LOAD_ENV = "AI_QA_EVAL_FORBID_MODEL_LOAD"


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def max_tokens(self) -> int: ...

    @property
    def fingerprint(self) -> str:
        """Identifies everything that determines the vectors."""
        ...

    def count(self, text: str) -> int:
        """Token count including special tokens, as the model would see `text`."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _check_limit(embedder: Embedder, texts: Sequence[str]) -> None:
    for i, text in enumerate(texts):
        tokens = embedder.count(text)
        if tokens > embedder.max_tokens:
            raise EmbedderError(
                f"input {i} has {tokens} tokens, over the {embedder.max_tokens}-token limit "
                f"of {embedder.name}; it would be truncated, so it is refused"
            )


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else vector


class HashingEmbedder:
    """Deterministic bag-of-words hashing. Test double; not a language model."""

    _WORD = re.compile(r"[a-z0-9]+")

    def __init__(self, *, dimensions: int = 256, max_tokens: int = 512) -> None:
        self.dimensions = dimensions
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "hashing-test-embedder"

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {"name": self.name, "dimensions": self.dimensions, "max_tokens": self._max_tokens}
        )

    def _words(self, text: str) -> list[str]:
        return self._WORD.findall(text.casefold())

    def count(self, text: str) -> int:
        return len(self._words(text)) + 2  # stand-ins for [CLS] and [SEP]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in self._words(text):
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        if not any(vector):
            vector[0] = 1.0
        return _normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        _check_limit(self, texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# --- the real model --------------------------------------------------------------


def load_tokenizer(path: Path) -> Tokenizer:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    # tokenizer.json may enable truncation or padding; both are turned off so every
    # count is exact and over-long input is caught by the limit check.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def load_onnx_session(path: Path, threads: int) -> Any:
    """Load the ONNX model: the only place the real model is ever loaded."""
    if os.environ.get(FORBID_MODEL_LOAD_ENV) == "1":
        raise EmbedderError(
            f"loading the real embedding model is forbidden ({FORBID_MODEL_LOAD_ENV}=1)"
        )
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def pinned_path(settings: EmbeddingSettings, root: Path, which: str) -> Path:
    """Path of a pinned model file, after checking it matches its SHA-256 pin."""
    if not settings.is_pinned:
        raise EmbedderError(
            f"{settings.name} is not pinned: set embedding.revision and the SHA-256 of "
            "embedding.model and embedding.tokenizer in config/rag.yaml (make model-fetch)"
        )
    pin = settings.model if which == "model" else settings.tokenizer
    path = settings.file_path(pin, root)
    if not path.is_file():
        raise EmbedderError(f"{path} is missing; run make model-fetch")
    actual = file_sha256(path)
    if actual != pin.sha256:
        raise EmbedderError(f"{path} has SHA-256 {actual}, pinned {pin.sha256}")
    return path


class WordPieceCounter:
    """Exact token counts from the pinned tokenizer, without loading the model."""

    def __init__(self, tokenizer: Tokenizer, max_tokens: int) -> None:
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens

    @classmethod
    def from_settings(cls, settings: EmbeddingSettings, root: Path) -> WordPieceCounter:
        return cls(load_tokenizer(pinned_path(settings, root, "tokenizer")), settings.max_tokens)

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True).ids)


class OnnxEmbedder:
    def __init__(self, settings: EmbeddingSettings, root: Path) -> None:
        self.settings = settings
        tokenizer = load_tokenizer(pinned_path(settings, root, "tokenizer"))
        self._counter = WordPieceCounter(tokenizer, settings.max_tokens)
        self._tokenizer = tokenizer
        pad_id = tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise EmbedderError("the pinned tokenizer has no [PAD] token")
        self._pad_id = pad_id
        self._session = load_onnx_session(pinned_path(settings, root, "model"), settings.threads)
        self._input_names = {i.name for i in self._session.get_inputs()}

    @property
    def name(self) -> str:
        return self.settings.name

    @property
    def max_tokens(self) -> int:
        return self.settings.max_tokens

    @property
    def fingerprint(self) -> str:
        return self.settings.fingerprint()

    @property
    def runtime(self) -> str:
        return f"onnxruntime {metadata.version('onnxruntime')}"

    def count(self, text: str) -> int:
        return self._counter.count(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        _check_limit(self, texts)
        vectors: list[list[float]] = []
        size = self.settings.batch_size
        for start in range(0, len(texts), size):
            vectors += self._embed_batch(texts[start : start + size])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self.settings.query_instruction + text])[0]

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        import numpy as np

        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=True)
        width = max(len(e.ids) for e in encodings)
        ids = np.full((len(encodings), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(encodings), width), dtype=np.int64)
        types = np.zeros((len(encodings), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            n = len(encoding.ids)
            ids[row, :n] = encoding.ids
            mask[row, :n] = encoding.attention_mask
            types[row, :n] = encoding.type_ids
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        missing = {"input_ids", "attention_mask"} - self._input_names
        if missing:
            raise EmbedderError(f"the ONNX model lacks inputs {sorted(missing)}")
        inputs = {name: array for name, array in feeds.items() if name in self._input_names}
        outputs = self._session.run(None, inputs)
        hidden = outputs[0]
        if hidden.ndim != 3 or hidden.shape[2] != self.settings.dimensions:
            raise EmbedderError(
                f"unexpected ONNX output shape {hidden.shape}; expected "
                f"(batch, tokens, {self.settings.dimensions})"
            )
        pooled = hidden[:, 0, :]  # CLS pooling, as bge models are trained
        if self.settings.normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return [[float(x) for x in row] for row in pooled]
