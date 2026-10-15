# 0006 — RAG corpus, chunking, index and assistant

**Status:** accepted (Phase 1a scope approved by the repo owner, 2026-10-12, with six
clarifications). Everything is built and has run on the real model. With the embedding
ranking alone, the retrieval smoke check failed its pre-set thresholds on 2026-10-13
(section lookups 7/12, paraphrases 11/16; minimum 0.8 each); see
`reports/retrieval_smoke/`. On 2026-10-15 the owner approved adding keyword and
section-number matching (see Retrieval), with the check's queries and thresholds
unchanged. With the hybrid ranking the check still failed on 2026-10-15: section
lookups 12/12 passed, paraphrases 12/16 (0.75) did not.

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
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0), used only by
  `retrieval.mode: hybrid_rerank`. It is pinned like the embedder (revision
  `233902d`, ONNX SHA-256 `5d3e70fd…`), downloaded by `make model-fetch` to `.cache/`
  and never committed. Its `tokenizer.json` is byte-identical to the embedder's.
- **Tests:** the default suite never loads either real model. `tests/conftest.py` sets
  a flag that makes the loader raise. Model tests run with `make test-model`.

## Retrieval
- **Why it changed:** the first real run used the embedding ranking alone and missed 5
  of 12 section lookups and 5 of 16 paraphrases. Most lookup misses cite a section
  number, which the embedding does not match on, and several returned a neighbouring
  section. The owner approved adding keyword and section-number matching on
  2026-10-15.
- **Modes:** `retrieval.mode` in `config/rag.yaml`. `dense` is the embedding ranking
  alone, as in the first run. `hybrid`, the default, is described below.
  `hybrid_rerank` is the candidate change described under "Held-out check" and is not
  the default; `make retrieval-smoke SMOKE_ARGS="--mode hybrid_rerank"` selects it for
  one run.
- **Keyword ranking:** BM25 over the same text that is embedded (heading path, label,
  text). The text is lowercased, a possessive "'s" is dropped, and the rest is split
  into letter and digit runs. Lucene's English stop words are removed and words are
  stemmed with the Snowball English stemmer (`snowballstemmer`, pinned). The keyword
  index is built in memory from the chunk manifest at start-up, so there is no second
  index to keep in step.
- **Fusion:** each ranking contributes its top 50 chunks. A chunk scores
  1 / (60 + rank) in each ranking it appears in (reciprocal rank fusion). Ties are
  broken by embedding similarity, then chunk ID.
- **Citations:** chunks that a question cites by section number (`11.10(e)`,
  `§ 11.3`, `III.C.4`) rank first: for each citation, its best-scoring chunk, then the
  other cited chunks by score. A citation covers the paragraphs under it, so a cited
  section with many paragraphs (§ 11.1 has 16) can fill the top k on its own. That is
  intended, because the question asks about that section. A regulation chunk also
  covers a finer citation inside it (the chunk §11.10(e) holds 11.10(e)(1)); a
  guidance chunk does not, because every numbered guidance heading is its own chunk.
  (Until `retrieval.version` 2 the guidance rule matched too, so the computer software
  assurance guidance's section III matched III.C.4 of the scope guidance.) Not
  recognized: a Roman
  numeral on its own, question numbers ("question 8") and document titles. The CFR
  title number is not checked, because the corpus holds one title.
- **Parameters:** BM25 k1 = 1.2 and b = 0.75 (Lucene's defaults), RRF k = 60 (Cormack,
  Clarke and Büttcher, 2009), and 50 candidates. These are published defaults, chosen
  and committed before the hybrid ranking ran on the smoke check, and not tuned on it.
  A test pins them.
- **Evidence:** each retrieval record holds the mode, a retrieval fingerprint (every
  setting except top_k, plus `retrieval.version`), the citations found, and for each
  chunk the fused score, embedding similarity and rank, BM25 score and rank, and the
  citation that matched it. With `hybrid_rerank`, records also hold the reranker's
  name and fingerprint, and each chunk's reranker score and the passage tokens cut to
  fit its 512-token limit. Run files are now schema 5 and smoke reports schema 3.
- **Limits of the smoke check:** the hybrid ranking was designed after seeing which
  smoke queries failed, so a pass on the same 28 queries is not independent evidence
  that it works on other questions. The citation rule targets the section-lookup kind
  directly. Independent evidence needs queries written after this change, ideally by
  the owner, such as the human-reviewed set planned for Phase 1b.

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
Run files are now schema version 5. The manifest records the prompt SHA-256 and the
corpus, chunk-manifest, embedding and retrieval fingerprints. Each answer record
holds its full retrieval (IDs, labels, text, scores and ranking details), so run
files do not depend on traces. Traces record the prompt version and the retrieved
chunk IDs and scores.

## Retrieval smoke check
`evals/dev_checks/retrieval_smoke.yaml` has 12 section lookups and 16 paraphrased
questions on Part 11 and the three guidance documents. They were written by the
developer and are not human-reviewed. Top-k (5) and the pass threshold (hit rate ≥ 0.8
for each kind) were fixed before any run. A query hits when a top-k chunk matches the
expected document and locator, or a locator nested under it. The check is kept apart
from the human-reviewed evaluation set and is never reported as an evaluation result.
The spec has not changed since the first run (SHA-256 `b70654fd…`), and neither has
the hit rule in `apps/rag/smoke.py`.

## Held-out check and the next change (fixed before either is run)
Approved by the owner on 2026-10-15, after the hybrid ranking failed the paraphrase
threshold.
- **Held-out questions:** `evals/dev_checks/retrieval_holdout.yaml`, with 6 section
  lookups and 15 paraphrases. Claude Code drafted them from the corpus text only, and
  no retrieval has been run on them. Their expected sections are ones the first check
  does not name. Top-k 5, the hit rule and the 0.8 thresholds are the same as the
  first check's. The owner reviewed and approved them unchanged on 2026-10-15,
  before their first run.
- **Candidate change:** cross-encoder reranking, `retrieval.mode: hybrid_rerank`. The
  hybrid ranking's top 50 chunks by fused score, plus any cited chunks, are scored by
  a cross-encoder that reads the question and the chunk's embedded text together.
  Cited chunks still rank first, then the rest by the reranker's score. The model is
  `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0) in ONNX form, pinned by
  revision and SHA-256 like the embedder, and run on one CPU thread. A pair over 512
  tokens loses the end of the chunk, and the number of tokens cut is recorded for each
  chunk.
- **Why this change:** for most of the hybrid paraphrase misses (P01, P08, P11, P14)
  the expected chunk was in the hybrid top 30, but the two rankings agreed on other
  chunks. A cross-encoder judges each question and passage pair directly. A larger
  embedding model was set aside because its expected gain is small. A reweighted
  fusion was set aside because its weight could only be chosen by fitting it to the
  first check's queries.
- **Decision rule:** after the owner's review, `hybrid` (the default) and
  `hybrid_rerank` each run once on both question sets. The candidate becomes the
  default only if it reaches every threshold on both sets. Otherwise the default stays
  `hybrid` and the results are reported. No setting changes between or after these
  runs without the owner's approval.

## Spending
Phase 1a makes no API calls. The generator alias (`sonnet`) is refused without
`--allow-paid`. No paid run is approved, including the optional live smoke.

## Open items
- **Retrieval smoke check:** fails on paraphrases with both rankings. The hybrid
  ranking fixed P09, P10, P13 and P16 but lost P01 and P14 (first by embedding, below
  50th by keyword, so fusion pushed them down) and P08. With the citation fix (run
  `20261015T010942Z-a03f3f`) only L11 changed, so paraphrases are still 12/16. The
  next step is the held-out check and the candidate change above, after the owner
  reviews the held-out questions.
- **Citation rule defect:** fixed in `retrieval.version` 2 (approved 2026-10-15). A
  guidance chunk no longer matches a finer citation, so section III of the computer
  software assurance guidance no longer matches III.C.4 (L11).
- **pre-commit size limit:** not needed. The largest source file is 752 KB, under
  the 1 MB limit.
