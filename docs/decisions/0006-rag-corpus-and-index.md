# 0006 — RAG corpus, chunking, index and assistant

**Status:** accepted (Phase 1a scope approved by the repo owner, 2026-10-12, with six
clarifications). Everything is built and has run on the real model. The retrieval
smoke check failed its pre-set thresholds on 2026-10-13 (section lookups 7/12,
paraphrases 11/16; minimum 0.8 each); see `reports/retrieval_smoke/`. The check is
unchanged pending the owner's review.

## Corpus
- **Sources:** 21 CFR Part 11 from the eCFR (as of 2026-10-08), plus three final FDA
  guidance documents: Part 11 Scope and Application (August 2003), Data Integrity and
  Compliance With Drug CGMP Q&A (December 2018) and Computer Software Assurance for
  Production and Quality Management System Software (February 3, 2026). Drafts are
  excluded (`status: final` is required). All are U.S. Government works; FDA's website
  policy states its content is in the public domain.
- **Manifest:** `data/corpus/sources.yaml` lists each document's title, kind
  (regulation or guidance), issuer, applicable domain, version label and date, eCFR
  reference with its as-of date and latest amendment, official source URL, retrieval
  date, SHA-256, size and the reuse notice that was checked. `SOURCES.md` is rendered
  from it, and a test fails if the two differ.
- **Frozen snapshot:** the official files are committed. Ingestion reads only them and
  stops if any file is missing or differs from its SHA-256. Refreshing is explicit
  (`make corpus-refresh`); it never happens during ingestion.
- **Fingerprint:** SHA-256 over each document's content hash and the fields that flow
  into chunks. Where and when a file was fetched is left out.

## Parsing
- **eCFR XML:** one unit per top-level paragraph, with its subparagraphs, labeled like
  `§11.10(e)`. Introductory text becomes `§11.10 (introductory text)`. A section with
  no paragraph markers is one unit. Nesting is inferred from the marker sequence, so
  `(i)` after `(h)` is a letter and `(i)` after `(1)` is a roman numeral.
- **Guidance PDF:** one unit per section. Headings come from the document's own
  table of contents (two of the three PDFs have no outline, and the third's outline
  is incomplete). Nesting follows the numbering (I. > A. > 1. > a., plus "(1)",
  "Appendix A", "Example 1:"), and each heading must match whole lines of the body,
  in order. Running line numbers (the 2003 guidance) are checked in sequence and
  removed. The banner, page numbers and TOC pages are removed. Footnotes are kept in
  reading order. Blocks are paragraphs within a page, so each chunk records its pages.
- **Failure:** unknown XML elements, out-of-sequence markers, TOC entries without a
  number or out of sequence, headings not found in the text and missing line numbers
  all stop the parse. Nothing is guessed or dropped silently.

## Chunking
- **Embedded text:** the heading path, the label and the text, one per line.
- **Token limit:** the whole string, special tokens included, must fit the model's
  512-token limit, counted with the pinned tokenizer (truncation off).
- **Splitting:** a unit that does not fit is split between blocks, then sentences,
  then words. Each part keeps the heading path and records its paragraph labels, pages
  and XML elements. Every finished chunk is counted again; one over the limit is an
  error. Nothing is truncated.
- **IDs:** `<doc_id>/<locator>`, plus `#<part>` when a unit is split.
- **Manifest:** `data/index/chunk_manifest.json` is committed and byte-identical for
  identical inputs. `make ingest-check` and a test verify that; the test never skips,
  so a missing or mismatched tokenizer fails offline CI.

## Embeddings and index
- **Model:** bge-small-en-v1.5 through onnxruntime (no PyTorch), with CLS pooling, L2
  normalization, the bge query instruction, and one CPU thread for repeatable vectors.
- **Pins:** `config/rag.yaml` pins the repo commit and the SHA-256 of the ONNX file
  and `tokenizer.json`. The embedder refuses to load while a pin is missing or a file
  differs. Weights are downloaded to `.cache/` and never committed.
- **Tokenizer:** committed in `data/models/bge-small-en-v1.5/` with the upstream MIT
  license and `SOURCE.md` (revision, URL, SHA-256), so CI can rebuild the manifest
  offline. It is byte-identical to upstream.
- **Index:** local Chroma (`data/chroma/`, not committed), cosine distance,
  ef 200. The collection records what it was built from: the embedding fingerprint,
  the chunk-manifest SHA-256 and the corpus fingerprint. The retriever refuses an index
  that does not match. Syncing upserts by chunk ID and deletes IDs no longer in the
  manifest, so repeated ingestion never duplicates a chunk. Before accepting an index,
  the retriever also checks every corpus file against its SHA-256, so an edited file
  is refused even when `sources.yaml` still describes the original.
- **Tests:** the default suite never loads the real model. `tests/conftest.py` sets a
  flag that makes the loader raise. Model tests run with `make test-model`.

## Assistant
- **Output:** `answer(question)` returns the behavior class (`answer`, `dont_know`,
  `refuse`; or `invalid` for output that breaks the contract, or `error` for a failed
  call), the answer, the cited chunk IDs as given, the citations that were not
  retrieved, the retrieved chunks with text and scores, the prompt version and
  SHA-256, the model call and the trace ID.
- **Stopping a run:** a provider error ends only its question (`behavior: error`).
  A stop for the whole run (budget exhausted, paid calls not allowed, interrupt)
  attaches the complete record to the exception (`stopped_run: true`: question,
  retrieval, prompt identity, resolved call settings, every failed attempt), where
  the run-file writer finds it with `answer_record_of`, and then propagates.
- **Prompts:** versioned files (`apps/rag/prompts/answer-v1.yaml`). A change makes a
  new file.
- **What a valid citation means:** it shows the answer points at a retrieved chunk.
  It does not show that the chunk supports the answer, or that the answer is correct.
  Unretrieved citations are flagged, never dropped.

## Evidence
Run files are now schema version 3. The manifest records the prompt SHA-256 and the
corpus, chunk-manifest and embedding fingerprints. Each answer record holds its full
retrieval (IDs, labels, text, scores), so run files do not depend on traces. Traces
record the prompt version and the retrieved chunk IDs and scores.

## Retrieval smoke check
`evals/dev_checks/retrieval_smoke.yaml` has 10 section lookups and 10 paraphrased
questions on Part 11, written by the developer and not human-reviewed. Top-k (5) and
the pass threshold (hit rate ≥ 0.8 for each kind) were fixed before any run. A query
hits when a top-k chunk matches the expected document and locator, or a locator
nested under it. The check is kept apart from the human-reviewed evaluation set and is
never reported as an evaluation result. Guidance queries will be added once the
guidance text is in the corpus, before the first run.

## Spending
Phase 1a makes no API calls. The generator alias (`sonnet`) is refused without
`--allow-paid`. No paid run is approved, including the optional live smoke.

## Open items
- **Retrieval smoke check failed** (see `reports/retrieval_smoke/`). Not changed yet;
  the next step is the owner's decision.
- **pre-commit size limit:** not needed. The largest source file is 752 KB, under
  the 1 MB limit.
