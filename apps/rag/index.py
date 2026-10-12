"""The local Chroma collection that holds one vector per chunk.

Chroma stores vectors and chunk IDs; the chunk manifest stays the source of truth
for chunk text and metadata. The collection's metadata records what it was built
from (embedding fingerprint, chunk-manifest SHA-256, corpus fingerprint), and the
retriever refuses a collection whose record does not match the current inputs.

Syncing is idempotent: chunks are upserted by their stable IDs and IDs no longer in
the manifest are deleted, so repeated ingestion never duplicates a chunk. Vectors
from a different embedding model are not comparable, so a change of embedding
fingerprint replaces the collection instead. While a sync runs, the collection
carries a "building" marker instead of a stamp, so an interrupted sync leaves an
index that queries refuse and the next sync rebuilds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from apps.rag.chunking import Chunk
from apps.rag.config import IndexSettings
from apps.rag.errors import StaleIndexError

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

# Collection metadata while a sync is in progress. It is not a valid stamp.
BUILDING = {"state": "building"}


@dataclass(frozen=True)
class IndexStamp:
    """What an index was built from. Stored as the collection's metadata."""

    embedder: str
    embedding_fingerprint: str
    chunk_manifest_sha256: str
    corpus_fingerprint: str

    def as_metadata(self) -> dict[str, str]:
        return {
            "embedder": self.embedder,
            "embedding_fingerprint": self.embedding_fingerprint,
            "chunk_manifest_sha256": self.chunk_manifest_sha256,
            "corpus_fingerprint": self.corpus_fingerprint,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> IndexStamp | None:
        if not metadata:
            return None
        try:
            return cls(**{k: str(metadata[k]) for k in cls.__dataclass_fields__})
        except KeyError:
            return None


@dataclass(frozen=True)
class SyncReport:
    upserted: int
    deleted: int
    replaced_collection: bool
    total: int


def _client(path: Path) -> ClientAPI:
    import chromadb
    from chromadb.config import Settings

    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False, allow_reset=False)
    )


class ChunkIndex:
    def __init__(self, path: Path, settings: IndexSettings) -> None:
        self.path = path
        self.settings = settings
        self._client = _client(path)

    def _get(self) -> Collection | None:
        from chromadb.errors import NotFoundError

        try:
            return self._client.get_collection(self.settings.collection, embedding_function=None)
        except NotFoundError:
            return None

    def _create(self) -> Collection:
        return self._client.create_collection(
            self.settings.collection,
            configuration={
                "hnsw": {
                    "space": self.settings.space,
                    "ef_construction": self.settings.ef_construction,
                    "ef_search": self.settings.ef_search,
                }
            },
            metadata=BUILDING,
            embedding_function=None,
        )

    def stamp(self) -> IndexStamp | None:
        collection = self._get()
        return IndexStamp.from_metadata(collection.metadata) if collection else None

    def sync(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], stamp: IndexStamp
    ) -> SyncReport:
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk is required")
        collection = self._get()
        replaced = False
        previous = IndexStamp.from_metadata(collection.metadata) if collection else None
        if collection is not None and (
            previous is None or previous.embedding_fingerprint != stamp.embedding_fingerprint
        ):
            self._client.delete_collection(self.settings.collection)
            collection, replaced = None, True
        if collection is None:
            collection = self._create()
        else:
            collection.modify(metadata=BUILDING)

        wanted = {c.id for c in chunks}
        stale = sorted(set(collection.get(include=[])["ids"]) - wanted)
        if stale:
            collection.delete(ids=stale)
        matrix = np.asarray(vectors, dtype=np.float32)  # Chroma stores float32
        batch = self._client.get_max_batch_size()
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            collection.upsert(
                ids=[c.id for c in part],
                embeddings=matrix[start : start + batch],
                documents=[c.text for c in part],
                metadatas=[
                    {"doc_id": c.doc_id, "label": c.label, "text_sha256": c.text_sha256}
                    for c in part
                ],
            )
        total = collection.count()
        if total != len(wanted):
            raise StaleIndexError(f"index holds {total} vectors for {len(wanted)} chunks")
        collection.modify(metadata=stamp.as_metadata())
        return SyncReport(len(chunks), len(stale), replaced, total)

    def query(
        self, vector: Sequence[float], k: int, expected: IndexStamp
    ) -> list[tuple[str, float]]:
        """Top-k (chunk ID, cosine similarity), best first. Refuses a stale index."""
        collection = self._get()
        if collection is None:
            raise StaleIndexError(f"no index at {self.path}; run make ingest")
        actual = IndexStamp.from_metadata(collection.metadata)
        if actual != expected:
            raise StaleIndexError(
                "the index was built from different inputs than the current configuration "
                f"(index: {actual}, expected: {expected}); run make ingest"
            )
        result = collection.query(
            query_embeddings=np.asarray([vector], dtype=np.float32),
            n_results=k,
            include=["distances"],
        )
        distances = result["distances"]
        if distances is None:
            raise StaleIndexError("the index returned no distances")
        pairs = list(zip(result["ids"][0], distances[0], strict=True))
        # Cosine distance = 1 - cosine similarity. Ties are broken by ID so the
        # order is reproducible.
        pairs.sort(key=lambda p: (p[1], p[0]))
        return [(chunk_id, 1.0 - float(distance)) for chunk_id, distance in pairs]
