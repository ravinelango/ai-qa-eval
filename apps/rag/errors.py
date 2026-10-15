"""Errors raised by the RAG pipeline. Each one stops the run with a reason; nothing is
skipped or truncated silently."""

from __future__ import annotations


class RagError(Exception):
    """Base class for RAG pipeline errors."""


class CorpusError(RagError):
    """The corpus snapshot is missing, unfrozen, or does not match its manifest."""


class SourceParseError(RagError):
    """A source document has a structure the parser does not recognize."""


class ChunkingError(RagError):
    """Text cannot be chunked within the token limit, or chunk IDs collide."""


class EmbedderError(RagError):
    """A local model (the embedder or the reranker) is unpinned, missing, does not match
    its pins, or got input it cannot take without truncation."""


class StaleIndexError(RagError):
    """The vector index was built from different inputs than the current configuration."""
