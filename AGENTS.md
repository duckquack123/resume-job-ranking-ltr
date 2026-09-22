# Do We Need LLM Re-Rankers? A Cost-Quality and Fairness Study of Feature-Based LTR for Resume-Job Matching

## Research Question
Can a feature-based LTR model match the quality of an LLM re-ranker for resume-job
matching at a lower cost, and does it exhibit different fairness properties?

## Hypotheses
- **H1 (Feature Groups):** The quality gap between LambdaMART and the LLM re-ranker
  shrinks as feature groups are added (lexical, dense, cross-encoder, structured).
- **H2 (Ambiguity):** Structured features help most on queries where embedding
  similarity is ambiguous.
- **H3 (Robustness):** LambdaMART is more stable across seeds and more robust across
  role families than the LLM re-ranker.
- **H4 (Fairness):** Rankings change measurably under counterfactual edits of
  protected-attribute proxies; model families differ in sensitivity; simple mitigations
  trade utility for lower sensitivity.

## Ground Rules
- Never fabricate numbers, citations or results; unknowns are marked TODO or UNKNOWN.
- Every reported number comes from a logged run and states n and a confidence interval.
- Labels must be independent of all model features; no labels from retrieval scores,
  title match or stated experience.
- Split by job and company; resume pools disjoint; near-duplicates removed across splits.
- All data stays local; no de-anonymization; no cloud annotation tools.
- Tune only on train/dev; the gold test set and TalentCLEF are never tuned on.
- Never run diagnostics on gold or dev jobs.

## Workflow Rules
- Stop after each step and write the review packet before proceeding.
- Do not start the next step until the user approves.
- Never combine steps.
- Batch questions with defaults; do not ask one at a time.
- Never modify frozen files (test_job_ids.txt, dev_job_ids.txt, dry_run_job_ids.txt,
  pilot_job_ids.txt, excluded_cv_ids.txt, jobs_dev.parquet, jobs_test.parquet,
  cvs_dev.parquet, cvs_test.parquet).

## Pointers
- `docs/status.md`: Current phase, approved decisions, rejected approaches, open items.
- `docs/decisions.md`: Dated log of all non-trivial decisions.
- `docs/review_gates.md`: Self-audit checklist required before every report.
- `docs/review_packet_template.md`: Template for submitting review packets.
