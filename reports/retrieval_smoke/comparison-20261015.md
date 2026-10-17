# Retrieval comparison, 2026-10-15: hybrid against hybrid_rerank on both question sets

Developer-authored plumbing checks, not evaluation results. Generated from the four JSON reports
linked below, each run once, as fixed in advance in ADR 0006 ("Held-out check and the next change").

**Decision rule:** the candidate (`hybrid_rerank`) becomes the default only if it reaches every
threshold (≥ 0.8 hit rate in the top 5, for each kind) on both sets.

**Outcome: the candidate does not become the default.** It passes the held-out set but fails the
original set's paraphrase threshold (12/16), so the default stays `hybrid`. No setting has been
changed after these runs.

| Ranking | Question set | Section lookups | Paraphrases | Overall | Report |
|---|---|---|---|---|---|
| `hybrid` | original 28 | 12/12 (1.00) pass | 12/16 (0.75) **FAIL** | FAIL | [`20261015T124236Z-05678b`](20261015T124236Z-05678b.json) |
| `hybrid_rerank` | original 28 | 12/12 (1.00) pass | 12/16 (0.75) **FAIL** | FAIL | [`20261015T124753Z-e3340a`](20261015T124753Z-e3340a.json) |
| `hybrid` | held-out 21 | 6/6 (1.00) pass | 13/15 (0.87) pass | PASS | [`20261015T124756Z-21951f`](20261015T124756Z-21951f.json) |
| `hybrid_rerank` | held-out 21 | 6/6 (1.00) pass | 15/15 (1.00) pass | PASS | [`20261015T125147Z-e6a49d`](20261015T125147Z-e6a49d.json) |

## Runs

- All four at code `478876e`, working tree clean, real models, no API calls.
- Original spec SHA-256 `b70654fda518…` (unchanged since the first run); held-out spec `b6d9b45387b8…`, owner-approved unchanged before these runs.
- Retrieval fingerprints: hybrid `52a8811310df…`, hybrid_rerank `c2003e5ee944…`. Chunk manifest `c1c2764b7d0c…` and embedding fingerprint `3e314946e74b…` as in every earlier run.
- Reranker: at most 24 passage tokens were cut from any chunk that reached a top 5.

## Original 28: hit rank per query

| ID | Kind | hybrid | hybrid_rerank | Expected |
|---|---|---|---|---|
| L01 | section_lookup | 1 | 1 | `21cfr11/11.10(e)` |
| L02 | section_lookup | 1 | 1 | `21cfr11/11.50` |
| L03 | section_lookup | 1 | 1 | `21cfr11/11.300` |
| L04 | section_lookup | 1 | 1 | `21cfr11/11.30` |
| L05 | section_lookup | 1 | 1 | `21cfr11/11.70` |
| L06 | section_lookup | 1 | 1 | `21cfr11/11.100(c)` |
| L07 | section_lookup | 1 | 1 | `21cfr11/11.200(a)` |
| L08 | section_lookup | 1 | 1 | `21cfr11/11.3(b)` |
| L09 | section_lookup | 1 | 1 | `21cfr11/11.1` |
| L10 | section_lookup | 1 | 1 | `21cfr11/11.10(a)` |
| L11 | section_lookup | 1 | 1 | `fda-part11-scope-2003/III.C.4` |
| L12 | section_lookup | 1 | 1 | `fda-data-integrity-2018/III.8` |
| P01 | paraphrase | **miss** | **miss** | `21cfr11/11.10(e)` |
| P02 | paraphrase | 1 | 1 | `21cfr11/11.50` |
| P03 | paraphrase | 3 | 1 | `21cfr11/11.100(a)`, `21cfr11/11.300(a)` |
| P04 | paraphrase | 1 | 1 | `21cfr11/11.300(c)` |
| P05 | paraphrase | 1 | 1 | `21cfr11/11.100(c)` |
| P06 | paraphrase | 2 | 1 | `21cfr11/11.70` |
| P07 | paraphrase | 1 | 1 | `21cfr11/11.10(i)` |
| P08 | paraphrase | **miss** | **miss** | `21cfr11/11.30` |
| P09 | paraphrase | 2 | 1 | `21cfr11/11.200(a)` |
| P10 | paraphrase | 4 | 2 | `21cfr11/11.10(b)` |
| P11 | paraphrase | **miss** | **miss** | `fda-part11-scope-2003/III.C.2` |
| P12 | paraphrase | 1 | 1 | `fda-part11-scope-2003/III.C.3` |
| P13 | paraphrase | 2 | 2 | `fda-data-integrity-2018/III.7` |
| P14 | paraphrase | **miss** | **miss** | `fda-data-integrity-2018/III.5`, `21cfr11/11.100(a)`, `21cfr11/11.300(a)` |
| P15 | paraphrase | 1 | 1 | `fda-csa-2026/V.A.2` |
| P16 | paraphrase | 3 | 1 | `fda-csa-2026/V.A.6` |

## Held-out 21: hit rank per query

| ID | Kind | hybrid | hybrid_rerank | Expected |
|---|---|---|---|---|
| HL01 | section_lookup | 1 | 1 | `21cfr11/11.10(k)` |
| HL02 | section_lookup | 1 | 1 | `21cfr11/11.2(b)` |
| HL03 | section_lookup | 1 | 1 | `21cfr11/11.300(e)` |
| HL04 | section_lookup | 1 | 1 | `fda-part11-scope-2003/III.B.2` |
| HL05 | section_lookup | 1 | 1 | `fda-csa-2026/V.A.4` |
| HL06 | section_lookup | 1 | 1 | `fda-data-integrity-2018/III.1.d` |
| HP01 | paraphrase | 1 | 3 | `21cfr11/11.10(f)` |
| HP02 | paraphrase | **miss** | 1 | `21cfr11/11.10(g)`, `fda-data-integrity-2018/III.4` |
| HP03 | paraphrase | 1 | 1 | `21cfr11/11.10(j)` |
| HP04 | paraphrase | 1 | 1 | `21cfr11/11.100(b)` |
| HP05 | paraphrase | **miss** | 1 | `21cfr11/11.200(b)` |
| HP06 | paraphrase | 2 | 3 | `21cfr11/11.300(b)` |
| HP07 | paraphrase | 4 | 2 | `21cfr11/11.300(d)` |
| HP08 | paraphrase | 1 | 1 | `fda-data-integrity-2018/III.4` |
| HP09 | paraphrase | 1 | 1 | `fda-data-integrity-2018/III.6` |
| HP10 | paraphrase | 1 | 1 | `fda-data-integrity-2018/III.10` |
| HP11 | paraphrase | 1 | 1 | `fda-data-integrity-2018/III.14` |
| HP12 | paraphrase | 1 | 4 | `fda-data-integrity-2018/III.15` |
| HP13 | paraphrase | 2 | 3 | `fda-part11-scope-2003/III.C.5`, `21cfr11/11.10(c)` |
| HP14 | paraphrase | 5 | 1 | `fda-csa-2026/V.A.1` |
| HP15 | paraphrase | 1 | 3 | `fda-csa-2026/V.A.3` |

## What the results show

- The four original paraphrases that miss with `hybrid` (P01, P08, P11, P14) also miss with `hybrid_rerank`. Reranking lifted five other original queries to rank 1 or 2, but not these.
- On the held-out set, `hybrid` already passes (13/15). `hybrid_rerank` fixed its two misses (HP02, HP05) and raised HP07 and HP14, but moved five hits lower within the top 5 (HP01, HP06, HP12, HP13 and HP15, from rank 1–2 to 3–4).
- So the original set's paraphrase failure is concentrated in four queries that neither ranking finds, while questions written afterwards pass with either ranking. That points at those four queries (wording or expectations) at least as much as at the ranking; that is a judgement for the owner, not something these runs settle.
