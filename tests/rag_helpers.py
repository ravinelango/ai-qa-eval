"""Builders for RAG tests. Everything here is synthetic test data: invented regulation
text, generated PDFs, a tiny tokenizer and an in-memory vector index."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from apps.rag.chunking import Chunk
from apps.rag.config import RagConfig
from apps.rag.index import IndexStamp

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "rag"
SYNTHETIC_ECFR = FIXTURES / "ecfr_part99_synthetic.xml"


def make_pdf(pages: Sequence[Sequence[str]], outline: Sequence[tuple[int, str, int]] = ()) -> bytes:
    """A PDF with one text line per entry; outline items are (level, title, 0-based page)."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    for lines in pages:
        page = writer.add_blank_page(612, 792)
        ops = ["BT", "/F1 10 Tf", "13 TL", "54 750 Td"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(ops).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
    parents: dict[int, Any] = {}
    for level, title, page_index in outline:
        parents[level] = writer.add_outline_item(title, page_index, parent=parents.get(level - 1))
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_tokenizer(path: Path, *, truncate_at: int | None = None) -> Path:
    """A small BERT-style WordPiece tokenizer ([CLS] text [SEP]) saved as tokenizer.json."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    words = ["the", "a", "of", "to", "and", "placeholder", "audit", "trail", "record", "log"]
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for word in words:
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.BertProcessing(("[SEP]", 3), ("[CLS]", 2))
    if truncate_at is not None:
        tokenizer.enable_truncation(max_length=truncate_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return path


class InMemoryIndex:
    """Exact cosine search over stored vectors; stands in for Chroma in unit tests."""

    def __init__(self, stamp: IndexStamp) -> None:
        self.stamp = stamp
        self.vectors: dict[str, list[float]] = {}

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.vectors[chunk.id] = list(vector)

    def query(
        self, vector: Sequence[float], k: int, expected: IndexStamp
    ) -> list[tuple[str, float]]:
        assert expected == self.stamp, "stale index"
        norm = math.sqrt(sum(x * x for x in vector))
        scored = [
            (cid, sum(a * b for a, b in zip(vector, v, strict=True)) / norm)
            for cid, v in self.vectors.items()
        ]
        scored.sort(key=lambda p: (-p[1], p[0]))
        return scored[:k]


ECFR_DOC: dict[str, Any] = {
    "doc_id": "synthetic-part99",
    "title": "Synthetic Part 99 (test fixture)",
    "kind": "regulation",
    "issuer": "Test fixture",
    "domain": "Synthetic ledgers (test only)",
    "status": "final",
    "version_label": "Synthetic eCFR snapshot",
    "format": "ecfr-xml",
    "file": "part99.xml",
    "reuse_notice": "Synthetic test data.",
    "ecfr": {"title": 1, "part": "99"},
}

GUIDANCE_DOC: dict[str, Any] = {
    "doc_id": "synthetic-guidance",
    "title": "Synthetic Guidance on Ledgers (test fixture)",
    "kind": "guidance",
    "issuer": "Test fixture",
    "domain": "Synthetic ledgers (test only)",
    "status": "final",
    "version_date": "2020-01-15",
    "version_label": "January 2020 (synthetic)",
    "format": "pdf",
    "file": "guidance.pdf",
    "source_url": "https://example.invalid/guidance.pdf",
    "reuse_notice": "Synthetic test data.",
}

GUIDANCE_PAGES: list[list[str]] = [
    ["Contains Nonbinding Recommendations", "Synthetic Guidance on Ledgers", "January 2020"],
    [
        "Contains Nonbinding Recommendations",
        "I. INTRODUCTION",
        "This placeholder guidance describes how to keep synthetic ledgers.",
        "II. DISCUSSION",
        "A. Change Logs",
        "Placeholder recommendation: keep a log of each change to a record,",
        "with the operator and the earlier value.",
        "2",
    ],
    [
        "Contains Nonbinding Recommendations",
        "B. Passwords",
        "Placeholder recommendation about password review and replacement.",
        "3",
    ],
]
GUIDANCE_OUTLINE: list[tuple[int, str, int]] = [
    (1, "I. INTRODUCTION", 1),
    (1, "II. DISCUSSION", 1),
    (2, "A. Change Logs", 1),
    (2, "B. Passwords", 2),
]


def frozen_entry(
    entry: dict[str, Any], data: bytes, *, as_of: str = "2026-01-02"
) -> dict[str, Any]:
    out = {
        **entry,
        "retrieved": "2026-01-05",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    if entry["format"] == "ecfr-xml":
        out["source_url"] = f"https://example.invalid/full/{as_of}/part99.xml"
        out["ecfr"] = {**entry["ecfr"], "as_of": as_of, "latest_amendment": "2025-06-01"}
    return out


def write_corpus(root: Path, documents: Sequence[tuple[dict[str, Any], bytes]]) -> RagConfig:
    """A frozen synthetic corpus under `root`, and a RagConfig that points at it."""
    corpus = root / "data" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry, data in documents:
        (corpus / entry["file"]).write_bytes(data)
        entries.append(frozen_entry(entry, data))
    (corpus / "sources.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "documents": entries}, sort_keys=False),
        encoding="utf-8",
    )
    return RagConfig.load(root=root)


def synthetic_corpus(root: Path) -> RagConfig:
    return write_corpus(
        root,
        [
            (ECFR_DOC, SYNTHETIC_ECFR.read_bytes()),
            (GUIDANCE_DOC, make_pdf(GUIDANCE_PAGES, GUIDANCE_OUTLINE)),
        ],
    )


def today() -> date:
    return date(2026, 1, 5)
