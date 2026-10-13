# Reports

Generated reports land here, each produced from a run whose raw output is saved in the
repo. No evaluation, judge-validation, red-team or A/B run has been run yet.

- `retrieval_smoke/` (Phase 1a): the developer-authored retrieval smoke check on the
  real index. A plumbing check, not an evaluation result. Run `20261013T132356Z-1795fb`
  failed both thresholds (section lookups 7/12, paraphrases 11/16; minimum 0.8).

Planned:
- `eval_runs/` and `judge_validation.md` (Phase 1)
- `redteam_findings.md` (Phase 2)
- `upgrade_decision_<date>.md` (Phase 3)
