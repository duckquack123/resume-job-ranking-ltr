# Project Status

**Current Phase:** Phase 2, Step C (gold labeling prep complete; awaiting approval to distribute annotation files). Judge tuning and Step 1 are frozen.

## Approved Decisions
- **Datasets:** Djinni English jobs and CVs (MIT license). TalentCLEF zero-shot only.
- **Pooling System:** BM25 + bge-small-en-v1.5 + e5-small-v2 pooling with K configurable, max 15 per job.
- **Annotation:** 0-3 rubric plus an Invalid flag; quadratic-weighted kappa.
- **Models:** LLM Judge: Llama-3.1-8B-Instruct. LLM Re-ranker: Qwen2.5-7B-Instruct.
- **Hardware:** A30 24GB.
- **Splitting:** 3 jobs per company cap in gold; separate train/dev/test CV pools; 27 dev jobs (21 eligible engineering; 6 dry-run + 15 pilot, of which 10 were hand-labeled), 72 test jobs (52 eligible, used as gold).
- **Data Handling:** No cloud-synced folders for annotator laptops. Only the pairs to be judged are exported locally. Files must be deleted after the pilot and main phases.
- **Annotation Export Design:** Whole jobs per annotator; separate files for (a) dry run/pilot, (b) main single-annotated, (c) main double-annotated; randomized order; stable pair IDs; provenance mapping kept in a separate file annotators never receive; no partner labels visible; no network calls; per-pair start/end timestamps recorded.
- **Adjudication Rule:** Compute agreement (QWK) on ALL double-annotated pairs before any discussion or dropping. Discuss every disagreement and record one consensus label. Single-annotated pairs are left unchanged.
- **Identical-Label Drop Rule:** Applied to evaluation only. After adjudication, for each job, if the final consensus labels for all its candidates are identical, drop those pairs and report the count. Agreement metrics are always reported on the full pre-drop set.
- **Role-Family Mapping:** Derived from Primary Keyword using a fixed lookup table (ROLE_MAP in data_pipeline.py). Used only for stratification and robustness splits, never as a feature or label. Unmapped keywords map to 'other'.
- **Evaluation:** Judged-pool evaluation plus condensed-list nDCG.
- **Tuning:** Judge-prompt tuning on 5 tuning jobs only (45 pairs). The 10 hand-labeled dev jobs (6 dry-run + 4 pilot, of the 21 eligible dev jobs) were split 5 tuning / 5 held-out (seed 42) before any judge output was seen. Held-out (41 pairs) was evaluated once.
- **Near-Duplicate Dedup:** The frozen splits were created with word-unigram dedup at threshold 0.85 (see decisions.md, 2026-09-20). A later cross-split pass (word 5-grams, 128 perms, threshold 0.70) removed items from the train side only; frozen dev/test IDs unchanged. The 5-gram pass is not the basis of the frozen files.

## Rejected Approaches (with reasons)
- **datasetmaster resumes:** Rejected due to unclear provenance/consent, mixed real and synthetic resumes, and domain mismatch with the Djinni jobs.
- **Synthetic resumes in main pool:** Real public resumes required; synthetic only as tagged augmentation to prevent unrealistic pooling.
- **Retrieval-score labels:** Any label source correlated with a model feature makes LambdaMART results circular.
- **Label Studio on login node:** Proposed in Plan v4, rejected. Replaced by purely local JSON task exports with no cloud sync.

## Role-Family Mapping (ROLE_MAP)
| Primary Keyword(s)                              | Role_Family     |
|-------------------------------------------------|-----------------|
| javascript, react, angular, vue                 | javascript      |
| java, spring                                    | java            |
| python, django                                  | python          |
| c#, .net                                        | .net            |
| php, laravel                                    | php             |
| c++, c                                          | c++             |
| ruby                                            | ruby            |
| go, golang                                      | go              |
| node.js                                         | node.js         |
| qa, manual qa                                   | qa              |
| qa automation                                   | qa automation   |
| devops, sysadmin                                | devops          |
| design, ui/ux                                   | design          |
| marketing, seo                                  | marketing       |
| project manager, product manager               | project manager |
| data science, data analyst, data engineer       | data            |
| sql, database                                   | sql             |
| security                                        | security        |
| (anything else)                                 | other           |

## Current Pool Sizes (Post-Dedup)
| Split | Jobs | CVs |
|-------|------|-----|
| Train | 137,893 | 168,137 |
| Dev   | 27 (frozen) | 21,019 (frozen) |
| Test  | 72 (frozen) | 21,020 (frozen) |

## Step 1 Execution Status (FROZEN)
- **Model Pipeline Frozen:** Feature list, LightGBM config, training target, seeds, fold file, judge labels, and trained model frozen and hashed.
- **Scale-Free Features:** 15 features (ratio-to-top, rank, and z-score within job). z-score matching claim dropped.
- **CV & Dev Metrics:** 5-fold CV NDCG@10 = 0.9221 (Random = 0.8510). Dev NDCG@10: Tuning = 0.7289, Held-out = 0.9056, All Dev = 0.8172.
- **Paired Comparisons (86 Human Pairs):** BGE (0.8353) > LambdaMART (0.8172) > Random (0.7593). LM vs Random paired diff = +0.0579, 95% CI: `[-0.0140, +0.1305]`.
- **Ablations & CV Length Correlation:** CV length Spearman with judge = 0.0358, with human = 0.0671.

## Step C Execution Status (COMPLETED)
- **Eligible Test Jobs (52 Jobs):** Evaluated from `jobs_test.parquet` (72 jobs) with blocklist and overrides (`134642_job` confirmed engineering).
- **Test Candidate Pooling ($K=3$):** BM25, BGE-small, E5-small on `cvs_test.parquet` (`114596_cv` excluded).
- **Within-Pool Dedup (5-gram Jaccard $\ge 0.70$):** 0 duplicates dropped; 446 unique candidate pairs across 52 jobs (22 multi-system overlaps).
- **Task Exports:**
  - Annotator A: 52 jobs, 446 pairs (`annotator_a_tasks.json`, SHA-256: `c1dd961afbc70e04b25a9ff8949b97a759cd57ccacfcd810048f7af853cc228b`).
  - Annotator B: 13 jobs, 115 pairs chosen by seed 42 stratified by role family (`annotator_b_tasks.json`, SHA-256: `28fc6a42bf9d180026e8961706fb80d7d20308e539d51a0ca6b831154129d8c3`; job IDs file SHA-256: `229cf6abbb9a88f4ce7bacc278a484a719b567cb20fd585a80760e33c53ab3c6`).
  - Provenance: Separate file `gold_provenance.json` (SHA-256: `fcf6544821d3f1b5d9d947e1f45bbccdc11c27b2a8e7e3a11295d866f8cfb69f`).
- **Annotation Interface:** `labeler_gold.html` (SHA-256: `e4b5ec6919541575d7ffdb556c1f4754ff7d1ffc33e1e8af67130598adeda271`) updated with independent `cannot_judge_job` and `is_invalid` flags, required dropdown reason selector, offline verification in headless Firefox (0 network calls), and session resume.
- **Overlap@5 Analysis:** Gold 52 jobs: union size 739 / 780 slots (14.21/job, multi-system overlap 5.26%). Overlap@5: BM25-BGE 3.46%, BM25-E5 1.54%, BGE-E5 11.54%, directly matching the dev-pool pattern (~3–5%).
- **Integrity:** `dev_hand_labeling_tasks.json` verified untouched (`2e6a5630e7b46e0e46257e58f7285d45cd5970497a67afd92e2800d14b449190`).
- **Next Phase:** Distribute annotation files to annotators upon user approval. Do NOT start Step A (cross-encoder) until gold labeling is launched.



## Open Items
- **Annotators:** [TODO]
- **Hours per week:** [TODO]
- **Scratch policy:** [TODO]
- **Deadline:** ~1 week (full scope restored)
- **Pytest full suite (6 tests on cleaned data):** PENDING

