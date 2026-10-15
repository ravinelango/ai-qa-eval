# Reports

Generated reports land here, each produced from a run whose raw output is saved in the
repo. No evaluation, judge-validation, red-team or A/B run has been run yet.

- `retrieval_smoke/` (Phase 1a): the developer-authored retrieval smoke check on the
  real index. A plumbing check, not an evaluation result. Run `20261013T132356Z-1795fb`
  (embedding ranking alone) failed both thresholds (section lookups 7/12, paraphrases
  11/16; minimum 0.8). Run `20261015T003855Z-e3adfd` (hybrid ranking) passed section
  lookups 12/12 but failed paraphrases 12/16. Run `20261015T010942Z-a03f3f` (hybrid,
  after the citation fix) changed only L11, now at rank 1; paraphrases are still 12/16.
  `comparison-20261015.md`: hybrid and hybrid_rerank, once each on the original and
  the owner-approved held-out set. Both pass the held-out set and fail the original
  set's paraphrases (12/16); the default stays hybrid.

Planned:
- `eval_runs/` and `judge_validation.md` (Phase 1)
- `redteam_findings.md` (Phase 2)
- `upgrade_decision_<date>.md` (Phase 3)
