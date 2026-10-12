# 0006 — RAG corpus, chunking, index and assistant

**Status:** accepted (Phase 1a scope approved by the repo owner, 2026-10-12, with six
clarifications). Code and offline tests are in place. Downloading the corpus and the
embedding model is blocked by the build environment's network policy.

## Corpus
- **Sources:** 21 CFR Part 11 from the eCFR, plus about three final FDA guidance
  documents (Part 11 scope and application, data integrity, computer software
  assurance). Drafts are excluded (`status: final` is required).
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
- **Guidance PDF:** one unit per section, with boundaries taken from the PDF outline.
  Page furniture is removed, and each block records its page.
- **Failure:** unknown XML elements, out-of-sequence markers, a missing outline and
  outline headings that are not in the text all stop the parse. Nothing is guessed
  or dropped silently.

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
  identical inputs. `make ingest-check` and a test verify that.

## Embeddings and index
- **Model:** bge-small-en-v1.5 through onnxruntime (no PyTorch), with CLS pooling, L2
  normalization, the bge query instruction, and one CPU thread for repeatable vectors.
- **Pins:** `config/rag.yaml` pins the repo commit and the SHA-256 of the ONNX file
  and `tokenizer.json`. The embedder refuses to load while a pin is missing or a file
  differs. Weights are downloaded to `.cache/` and never committed.
- **Index:** local Chroma (`data/chroma/`, not committed), cosine distance,
  ef 200. The collection records what it was built from: the embedding fingerprint,
  the chunk-manifest SHA-256 and the corpus fingerprint. The retriever refuses an index
  that does not match. Syncing upserts by chunk ID and deletes IDs no longer in the
  manifest, so repeated ingestion never duplicates a chunk.
- **Tests:** the default suite never loads the real model. `tests/conftest.py` sets a
  flag that makes the loader raise. Model tests run with `make test-model`.

## Assistant
- **Output:** `answer(question)` returns the behavior class (`answer`, `dont_know`,
  `refuse`; or `invalid` for output that breaks the contract, or `error` for a failed
  call), the answer, the cited chunk IDs as given, the citations that were not
  retrieved, the retrieved chunks with text and scores, the prompt version and
  SHA-256, the model call and the trace ID.
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
- **Network access:** `www.ecfr.gov`, `www.fda.gov` and `huggingface.co` (and the
  CDN host it redirects downloads to) must be allowed before the corpus, pins,
  manifest and smoke result can exist.
- **Committing `tokenizer.json`:** about 0.7 MB, MIT/Apache licensed. Committing it
  would let CI check the committed manifest; otherwise that test skips in CI. It is
  not model weights.
- **pre-commit size limit:** the 1 MB limit in pre-commit would be raised for
  `data/corpus/` only, if a source file exceeds it.
