# Decision Log

- **2026-09-19:** Adopted Djinni English jobs/CVs. Rejected datasetmaster (unclear provenance/consent, mixed real and synthetic resumes, domain mismatch with Djinni jobs).
- **2026-09-19:** Synthetic resumes restricted to tagged augmentation to prevent unrealistic pooling.
- **2026-09-19:** Decided on independent labels (human/LLM judge) rather than retrieval scores to prevent circularity.
- **2026-09-20:** Adopted 3 independent pools for CVs (Train/Dev/Test) to prevent any leakage of few-shot or development CVs into the evaluation set.
- **2026-09-20:** Cap Test jobs at 3 per company and stratify by Role Family to prevent single-company dominance (e.g., Gemicle).
- **2026-09-20:** Rejected Label Studio over SSH on the login node (which was proposed in Plan v4). It was replaced by purely local JSON task exports to comply with offline data handling rules.
- **2026-09-20:** Frozen test_job_ids.txt (72 IDs), SHA-256: `4b282bcb9df5ca0d9d65915eb5567a6f83998e6e684f5383c321af05e950e95f`
- **2026-09-20:** Frozen dev_job_ids.txt (27 IDs), SHA-256: `369969c016eeb3a4e9b693a1e002df9692e72909d72a1fda23eb115c42e6b308`
- **2026-09-20:** Frozen dry_run_job_ids.txt (first 10 of 27 dev IDs, sorted): SHA-256: `7b9de57361201c136a381bcc29ee53971b319af0faa927a96bb80f5d40c02469`
  IDs: 105723_job, 110001_job, 129445_job, 140822_job, 141140_job, 17637_job, 19780_job, 21604_job, 22047_job, 26348_job
- **2026-09-20:** Frozen pilot_job_ids.txt (remaining 17 dev IDs, sorted): SHA-256: `c778f0c43f43fe2f1fa466e7cccb11705b26b7c14848cbbc3b1754a58866e036`
  IDs: 31500_job, 34898_job, 35178_job, 39885_job, 46596_job, 49204_job, 50223_job, 53525_job, 64381_job, 70278_job, 75798_job, 79686_job, 86405_job, 8660_job, 87154_job, 92624_job, 9423_job
- **2026-09-20:** Cross-split near-duplicate dedup applied at threshold 0.70 (word 5-grams, 128 perms). Train side only; dev/test frozen IDs and hashes unchanged.
  - Jobs removed from train: 1 (train/17091_job ↔ test/17502_job, Jaccard 0.773). Train: 137894 → 137893.
  - CVs removed from train: 16 (train_vs_dev: 10 unique train IDs, train_vs_test: 6 unique train IDs, with overlap counted once). Train: 168153 → 168137. Dev/test pair found: 1 (no action; frozen).
  - Dev vs Test job pairs: 0. Dev vs Test CV pairs: 1 (no action; frozen IDs unchanged).
  - Updated jobs_train.parquet SHA-256: `344268770c61b123a5a3e1a8160f6a022fdb3eb1069f010b70e1eaeffd106bb6`
  - Updated cvs_train.parquet SHA-256: `42ca79d5a92aa6c9f12f33d266f5759674d27e550e75bfc60dde51e883e5c60c`
- **2026-09-20:** excluded_cv_ids.txt created (1 ID: 114596_cv — dev/test near-duplicate, Jaccard 0.719). SHA-256: `28a6fdce10850af36edb84623d7b45de370c0e374fd40daacbcdb23ef6793610`. Pooling scripts must skip this ID; CV cross-pool test applies exclusion before checking.
- **2026-09-20:** Annotation eligibility rule: a job is eligible if its Primary Keyword does NOT map to a non-engineering domain. Non-engineering blocklist: {hr, sales, support, recruiter, business analyst, lead generation, scrum master, salesforce, artist, legal, finance, accounting, copywriter, content, invoicing, engagement manager, growth, business development, delivery manager, vfx, project manager, marketing, product manager}. 'other' jobs with engineering keywords (android, ios, unity, flutter, scala, rust, kotlin, etc.) are ELIGIBLE. Test set: 59 eligible / 72 total. Dev set: 21 eligible / 27 total. Since 59 < 60, an extra test set of ~20 jobs will be drawn from train (fixed seed 42, stratified by role family, removed from train, hashed — to be approved before implementation).
- **2026-09-20:** Split dev into 6 dry-run and 15 pilot jobs (seed 42, stratified by role family). The earlier 10/17 lists were never used because they contained ineligible non-engineering jobs. 
  - `dry_run_job_ids.txt` SHA-256: `07fdee88e840c36e1da893c34ab78e2fcb161677dd93cfa45a7b1227c2841594`
  - `pilot_job_ids.txt` SHA-256: `72a764c2ea5952f28c7a858fd1fd575895d5b3c8a97489416308963175978ced`
- **2026-09-20:** Dedup clarification: the frozen splits (jobs_train/dev/test, cvs_train/dev/test) were created using the original **word unigram** deduplication at threshold 0.85 (which dropped 3,904 jobs and 58 CVs). The word-5-gram (threshold 0.70) results computed later are an independent audit/check, not the basis of the frozen files.
- **2026-09-20:** Positive control arithmetic corrected (101 shingles on both sides, 3 non-overlapping). Exact Jaccard is 98/104 ≈ 0.942. The 128-permutation MinHash estimate for this pair is 0.9453, well above the 0.70 threshold.
- **2026-09-20:** Documented Limitation: `Role_Family` stratification uses the `Primary Keyword` via a fixed mapping, which is noisy (e.g., "Angular Developer" -> java, "Solidity Developer" -> java). This is accepted as a limitation since `Role_Family` is only used for stratification, not as a model feature or label.

## Label Acquisition Strategy (Phase 2 - Learning Mode)
- **LLM Judge (Planned Route):** Best fit. Scalable, and labels are independent of features.
- **Hand Labels:** Too small for training (e.g., 86 pairs), but essential for checking the judge.
- **Rules from data (keyword match, experience, skill overlap):** REJECTED. These labels are computed from the same signals features use; LambdaMART would just relearn the rule.
- **Click or application logs:** REJECTED. Not available in the Djinni data.
- **Synthetic queries (InPars/Promptagator):** OPTIONAL alternative for later. Gives binary labels and generated job text looks different from real postings.

## Tuning vs Held-out Split (Phase 2)
- **2026-09-20:** The 10 dev hand-labeling jobs (6 dry-run + 4 pilot) were split by seed 42 into 5 tuning jobs and 5 held-out jobs, BEFORE looking at any judge output.
  - Tuning jobs: 105723, 110001, 26348, 53525, 75798
  - Held-out jobs: 129445, 22047, 50223, 64381, 70278
  - Hashes: tuning_job_ids.txt (`cda82db079b56722656df5fec3c973467801577e7d0b88ea6cc64fc2ad80c1c6`), heldout_job_ids.txt (`461714be770353f2eb78d51899c86e1df3d5e809f69a9a40fcaa55f73ab9277c`).

## Llama 3.1 Licence Review
- **2026-09-20 (Corrected 2026-09-21):** Corrected summary of Section 1.b.i of the Llama 3.1 Community License Agreement. Under Section 1.b.i, if Llama 3.1 outputs or materials are used to create, train, fine-tune, or improve an AI model that is distributed or made available, the resulting model name MUST begin with "Llama" (followed by a distinguishing name). In addition, distribution requires prominently displaying "Built with Llama", providing the Notice file and a copy of the License Agreement, and strictly complying with Meta's Acceptable Use Policy.

### Tuning/Held-out Split
- `data/processed/tuning_job_ids.txt`: cda82db079b56722656df5fec3c973467801577e7d0b88ea6cc64fc2ad80c1c6
- `data/processed/heldout_job_ids.txt`: 461714be770353f2eb78d51899c86e1df3d5e809f69a9a40fcaa55f73ab9277c
- **Limitation:** The split of the 10 dev jobs into 5 tuning and 5 held-out jobs (seed 42) was done randomly. It is not stratified by role family, which may result in imbalanced coverage for certain roles.

### Llama 3.1 License Terms
- **Source:** https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/blob/main/LICENSE
- **Date read:** Sept 21, 2026
- **Summary:** Under Section 1.b.i, if Llama 3.1 outputs or materials are used to create, train, fine-tune, or improve an AI model that is distributed or made available: (1) the model name MUST begin with "Llama" (followed by a distinguishing name), (2) any distribution must prominently display "Built with Llama", (3) distribution must include the Notice file and a verbatim copy of the License Agreement, and (4) use must comply with Meta's Acceptable Use Policy.
- **Project Impact:** If we distribute or make available our trained LambdaMART model (or any model trained on Llama 3.1 outputs), its name must begin with "Llama" (e.g., `Llama-LambdaMART` or `Llama-Ranker`), and we must package the Notice file, license agreement copy, "Built with Llama" attribution, and follow the Acceptable Use Policy.

## 500-Job Training Set & Sampling (Phase 2)
- **2026-09-21:** The earlier 500-job set (`train_500_job_ids.txt`, SHA-256: `1ef137e91705101f9da59dfe22c278e929043beb2aed8394ffba81d261609407`) was **NEVER USED** for training or judging because it violated the engineering eligibility sampling rule (it included 28 design, 4 marketing, and 84 generic other jobs).
- **2026-09-21:** A new, compliant 500-job training set was generated by enforcing the full test-set blocklist (excluding design, marketing, project manager, hr, sales, etc.) and filtering "other" to only valid engineering keywords (`android`, `ios`, `unity`, `flutter`, `scala`, `rust`), while retaining max 3 per company, role-family stratification, seed 42, and near-duplicate removal (word 5-gram, 0.70).
  - Frozen `train_500_job_ids.txt` SHA-256: `4d8190babd8b314bc58bb3cfd19930ef9274e50e064849abf6652537f3345c20`
  - Frozen `train_500_cv_folds.json` SHA-256: `f4c35fd5095ca45e506d05d4bfb99434e0bbebb6083b94f78d20ec51b48be61f`
  - Frozen `train_500_pairs.parquet` SHA-256: `8667a3e28b31535073a3f236bf143655f7f2af68b2612d4781015a9365b393b6`
  - Total candidate pairs pooled: 4,369 across 500 jobs. Candidate CV near-deduplication (word 5-gram, 0.70) within each job pool removed 0 near-duplicates (all 4,369 pairs unique).

## Prompt Template Evolution & Hashes
- **2026-09-20 (Template v1 - Raw Prompt):** Plain string ending with `Score:`. SHA-256: `e62f352e51f837eae4eaf1bf86b3cd1c9817240d3faaa40c8ed887c3db1fcc28`. Suffered from ~3% digit probability mass because the model predicted a space (token 220) or conversational continuation directly after the colon.
- **2026-09-21 (Template v2 - Chat Format with Assistant Prefix):** Wrapped user text in Llama 3.1 Instruct chat template (`<|start_header_id|>system...<|start_header_id|>user...`) and ended prompt with assistant-turn prefix `<|start_header_id|>assistant<|end_header_id|>\n\nGrade (0-3): `. SHA-256: `76d3f32cd7eb2c9282234dc8f370f3b7c1650203ebfffe1de71d074da43f15fc`. Fixed token boundary; digit mass increased to 0.9508 (mean) and 0.8981 (p5).
- **Token Length Accounting:** The shift in unpadded mean prompt length from ~798 to 937.1 tokens was caused by: (1) different pairs evaluated (`generate_timing_tasks.py` pairs had mean 797.9 tokens vs `prepare_timing_tasks.py` pairs with mean 898.1 tokens on the raw template), and (2) chat template overhead (+39 net tokens per prompt from system/turn headers and assistant prefix, bringing mean length from 898.1 to 937.13).
- **Template Diff:**
```diff
--- Template v1 (e62f352e...)
+++ Template v2 (76d3f32c...)
+<|start_header_id|>system<|end_header_id|>
+
+Cutting Knowledge Date: December 2023
+Today Date: 26 Jul 2024
+
+You are an expert technical recruiter evaluating candidates for software engineering jobs.<|eot_id|>
+<|start_header_id|>user<|end_header_id|>
+
 [Job Description & Candidate CV Text]
-Score:
+<|eot_id|><|start_header_id|>assistant<|end_header_id|>
+
+Grade (0-3): 
```

## Canonical Timing & Sanity Check Dataset
- **2026-09-21:** Fixed and frozen ONE single 100-pair timing dataset: `data/processed/timing_tasks.json` (SHA-256: `f372e7e276c259775a485d75e9df909c81591532a52a024c48ce143abdf42a9c`). All subsequent throughput, digit mass, order-swap, and format sanity tests must strictly use this single frozen file.

## 500-Job Set Refinements & Pair-Level Exclusion
- **2026-09-21:** Four non-engineering jobs under the "unity" keyword (`11258_job`, `39848_job`, `83543_job`, `135420_job`) were identified. The frozen job IDs file `train_500_job_ids.txt` was preserved unchanged. Instead, a hashed exclusion file `data/processed/excluded_train_job_ids.txt` (SHA-256: `0b6e1ceb82092a90b72cb9003130150025a2d009db2d0ef0fc189b111ae5c62a`) was created and applied at pair level.
- Excluded 33 candidate pairs, reducing training pairs from 4,369 to **4,336**.
- Clean pairs file: `data/processed/train_500_pairs_clean.parquet` (SHA-256: `a4b8ba753a413ce0ed88f28ca0a915fbc79f4313abdd9d0e73918e4460412da4`).

## Prompt Tuning on 45 Tuning Pairs (Phase 2)
- **2026-09-21:** Prompt tuning executed exclusively on the 45 tuning pairs (`labels_tuning.json`, SHA-256: `36cae5c60a3230cf644fca74bdf642a1f36f3d06fa27202ba56e121b9c61a6b6`) across 4 rubric iterations:
  1. `v1_baseline` (SHA-256: `cec806906ebf6b25fec267f7e2fc969667b96ccf704481c68290976ac2540f5a`) -> Tuning QWK: **0.4375**, Spearman: 0.5732, Exact Agreement: 33.3% (15/45).
  2. `v2_tech_stack_clarity` (SHA-256: `373140e18a860632b6bda3a8d3f621b42c95ab821d8f56d19d38f3ba3073f1a7`) -> Tuning QWK: **0.4020**, Spearman: 0.5261, Exact Agreement: 33.3% (15/45).
  3. `v3_competency_alignment` (SHA-256: `aff2f962b627662b27c4529529be8e17d0bfae1e6b6ecadcab1da6b3341d5b4c`) -> Tuning QWK: **0.4951**, Spearman: 0.4979, Exact Agreement: 48.9% (22/45).
  4. `v4_seniority_requirements` (SHA-256: `f868bdbedbe900e7d5243a068079f7c5c0d4efd88110e24a0b82df3740efeed6`) -> Tuning QWK: **0.5517**, Spearman: 0.5681, Exact Agreement: 48.9% (22/45).
- **Frozen Canonical Judge Prompt:** `v4_seniority_requirements` selected as the best and frozen.
  - Template SHA-256: `f868bdbedbe900e7d5243a068079f7c5c0d4efd88110e24a0b82df3740efeed6`
  - Saved to: `data/processed/frozen_judge_prompt_template.txt`
  - Held-out jobs and labels remain completely untouched.

## Judge Aggregation Rule (Order-Averaging)
- **2026-09-21:** Based on order-swap analysis showing 39% grade flips under prompt sequence reversal (CV first vs Job first), the final judge scoring rule is frozen as the average of expected scores from both prompt orders:
  $$\text{Expected Score}_{\text{final}} = \frac{\mathbb{E}[\text{Score}_{\text{Job}\to\text{CV}}] + \mathbb{E}[\text{Score}_{\text{CV}\to\text{Job}}]}{2}$$
  This is an aggregation rule, not a prompt version. On the 45 tuning pairs, order averaging lifts QWK from 0.5517 to **0.6593**, Spearman correlation from 0.5681 to **0.6486**, exact agreement from 48.9% to **51.1%**, and within-one-grade rate from 82.2% to **88.9%**.

## One-Time Held-Out Evaluation (Phase 2 Check)
- **2026-09-22:** Executed one-time evaluation of frozen prompt v4 with order-averaging on the 41 held-out pairs (`labels_heldout.json`, SHA-256: `5f852d903f82464b8d1b95d5588ac7ca7050693623c23c798097db47f077201d`).
  - Results: QWK **0.3734** (single-order: 0.3972), Spearman **0.5991** (single-order: 0.5429), Exact Agreement **31.7% (13/41)**, Within-one-grade rate **85.4% (35/41)**. Mean per-job NDCG@10: **0.9200** (vs expected random permutation NDCG@10 baseline of **0.8396**). Mean per-job Spearman: **0.3513**.
  - Strict discipline: Zero prompt modifications made following evaluation. All prompt templates remain frozen.

## Step 1 Execution (Phase 2 - Feature LTR Pipeline)
- **2026-09-22:** Non-engineering jobs exclusion from 500-job training set:
  - 6 jobs excluded (`2070_job`, `28117_job`, `30297_job`, `48467_job`, `88628_job`, `140972_job`) along with 4 Unity non-engineering jobs (`11258_job`, `39848_job`, `83543_job`, `135420_job`).
  - Preserved frozen ID file `train_500_job_ids.txt` unchanged. Created hashed exclusion file `data/processed/excluded_train_job_ids_v2.txt` (SHA-256: `97839027cfc63c8c37d2eb92b9b889159f3ebf7c47c1f0e3317d9a4b2a248cfd`).
  - Total candidate pairs pooled: 4,369 -> 4,336 -> **4,283 pairs** across 490 jobs (53 pairs dropped for the 6 jobs).
  - Clean training pairs file: `data/processed/train_490_pairs_clean.parquet` (SHA-256: `af6623a6a3e4f0dfbd6cd99ef54e008c73f36ceefaec01c6ba5af8c6da5bd6ad`).
- **2026-09-22:** Held-Out Bootstrap Confidence Intervals ($n = 41$ pairs, 5-job cluster):
  - Spearman: **0.5991**, 95% Pair Bootstrap CI: `[0.3245, 0.7938]`.
  - QWK: **0.3734**, 95% Pair Bootstrap CI: `[0.0797, 0.6113]`.
  - Saved to `data/processed/results/heldout_diagnostics.json`.
- **2026-09-22:** Features Extracted for 4,283 Train Pairs & 86 Dev Pairs:
  - Exact 15 scale-free features: within-job z-score, within-job ratio to top candidate, and rank for BM25, BGE, E5, CV length, and tech overlap. Replaced all raw-scale differences (`diff_top`) with scale-free z-scores and ratios.
  - Documented BM25 corpus shift: Train pool ($N=168,137$) raw BM25 mean = 42.51 $\pm$ 20.09 vs Dev pool ($N=21,019$) raw BM25 mean = 190.51 $\pm$ 99.89. Within-job transformations mitigate cross-split scale mismatch, though distribution differences remain from pooling and document lengths.
  - Feature files: `data/processed/features_train_scale_free.parquet` and `data/processed/features_dev_86_scale_free.parquet`.
- **2026-09-22:** First-Stage Retriever Evaluation on 86 Dev Hand Labels:
  - BGE-small: NDCG@10 = **0.8353** (95% CI `[0.7435, 0.9022]`), MRR ($\ge 2$) = **0.8083** `[0.6000, 1.0000]`.
  - E5-small: NDCG@10 = **0.7524** (95% CI `[0.6607, 0.8419]`), MRR ($\ge 2$) = **0.7083** `[0.5333, 0.9000]`.
  - Random Baseline: NDCG@10 = **0.7593** (95% CI `[0.6994, 0.8174]`), MRR ($\ge 2$) = **0.7198** `[0.6334, 0.8106]`.
  - BM25: NDCG@10 = **0.7337** (95% CI `[0.6345, 0.8314]`), MRR ($\ge 2$) = **0.5733** `[0.3550, 0.7834]`.
  - Provenance Mapping: **BGE retrieved the candidates scored highest** by human annotations (mean grade 1.93; 12/25 Grade 3 candidates; 63.3% relevant).
- **2026-09-22:** Step 1 Revisions & Protocol Clarifications:
  - Folds: `train_500_cv_folds.json` (frozen, company-grouped 5 folds) was restricted to 490 jobs by excluding the 10 flagged IDs, without regeneration.
  - LightGBM Hyperparameters: `objective='lambdarank'`, `metric='ndcg'`, `eval_at=[10]`, `lr=0.05`, `num_leaves=15`, `min_child_samples=10`, `n_jobs=4`.
- **2026-09-22:** Step 1 Pipeline Freeze (Pre-Gold Lock):
  - In accordance with research protocol, all model components, hyperparameters, features, and targets are frozen prior to receiving gold test labels:
    - **Feature List (15 scale-free features):** `data/processed/frozen_feature_list.json` (SHA-256: `aec559529a37ffff9bfe2bdeb700b2abd807b4ffab55b9dd12a9c9026ec0d339`).
    - **LightGBM Configuration:** `data/processed/frozen_lgbm_config.json` (SHA-256: `d27861698af9dd2223d8e8aadfae6ac8d392c599dca184e178ae2d761c8b0a65`). Hyperparameters (`lr=0.05`, `num_leaves=15`, `min_child_samples=10`, `n_estimators=100`, early stopping 15 rounds) were chosen strictly via 5-fold company-grouped cross-validation on the 490 training jobs' judge labels, never on human labels.
    - **Training Target:** Rounded average expected grades (0, 1, 2, 3) from `train_judge_results.jsonl`.
    - **Seeds:** Fixed seed 42.
    - **Fold File:** `data/processed/train_500_cv_folds.json` (SHA-256: `f4c35fd5095ca45e506d05d4bfb99434e0bbebb6083b94f78d20ec51b48be61f`), restricted via `excluded_train_job_ids_v2.txt` (SHA-256: `97839027cfc63c8c37d2eb92b9b889159f3ebf7c47c1f0e3317d9a4b2a248cfd`).
    - **Judge Label File:** `data/processed/train_judge_results.jsonl` (SHA-256: `d3c065425f1492e26835fda32020c13b541812a605b1e4826ee74f008eadc519`).
    - **Trained Model Booster:** `models/lambdamart_step1_frozen.txt` (SHA-256: `6f5d6053eb8456e0baa3ee66f9068ba978f44cbfb0119c20e9e941f9fdb6b313`, 100 trees).
  - **Single Evaluation Rule:** Gold labels will be evaluated strictly ONCE, after upload. No post-hoc tuning.

- **2026-09-22:** Step C: Gold Test Set Generation & Task Exports:
  - **Eligible Test Jobs (52 Jobs):** Evaluated from `jobs_test.parquet` (72 jobs) via `is_eligible` blocklist and `eligibility_overrides.csv`.
  - **134642_job Decision:** Job `134642_job` ("TDM with focus on QA\QC (VR-61299)" at Luxoft) was explicitly verified and approved as eligible engineering. While generic "delivery manager" is blocklisted, this role has an explicit technical QA Automation / Test Delivery Management focus, documented in `eligibility_overrides.csv`.
  - **Candidate Pooling ($K=3$):** Top-3 candidates pooled from BM25, BGE-small-en-v1.5, and E5-small-v2 on `cvs_test.parquet` with `excluded_cv_ids.txt` applied (`114596_cv` excluded).
  - **Deduplication:** Word 5-gram candidate deduplication within each job's pool (threshold 0.70) found 0 near-duplicates (all 446 pairs unique). Total retrieval slots = 468; unique pairs = 446 (22 multi-system overlaps; 8.58 candidates/job).
  - **Task File Separation:**
    - **Annotator A (52 jobs, 446 pairs):** `data/processed/annotator_a_tasks.json` (SHA-256: `c1dd961afbc70e04b25a9ff8949b97a759cd57ccacfcd810048f7af853cc228b`).
    - **Annotator B (13 jobs, 115 pairs):** Sampled with seed 42 stratified by role family (3 js, 3 qa automation, 2 other, 1 java, 1 devops, 1 ruby, 1 python, 1 php). Hashed before labeling: `data/processed/annotator_b_job_ids.txt` (SHA-256: `229cf6abbb9a88f4ce7bacc278a484a719b567cb20fd585a80760e33c53ab3c6`). Task file: `data/processed/annotator_b_tasks.json` (SHA-256: `28fc6a42bf9d180026e8961706fb80d7d20308e539d51a0ca6b831154129d8c3`).
    - **Provenance File:** Kept strictly separate; annotators never see retrieval provenance. `data/processed/gold_provenance.json` (SHA-256: `fcf6544821d3f1b5d9d947e1f45bbccdc11c27b2a8e7e3a11295d866f8cfb69f`).
  - **Annotation Interface:** Created `data/processed/labeler_gold.html` as an offline single page with 3 Strong / 2 Good with minor gaps / 1 Weak / 0 Not relevant / Invalid buttons, on-screen rubric and ignore rules, per-job notes box, pairs grouped by job and shuffled within each job, a "cannot judge this job" flag, start/end timestamps, and autosave/resume. Zero network requests.
  - **Dev Hand Labeling Integrity:** `dev_hand_labeling_tasks.json` verified 100% UNTOUCHED (SHA-256: `2e6a5630e7b46e0e46257e58f7285d45cd5970497a67afd92e2800d14b449190`).
  - **Planned Gold Analysis:** NDCG@3, NDCG@10, and MRR ($\ge 2$) with company-cluster bootstrap CIs and random permutation baselines.
- **2026-09-22:** Step C: Gold Pooling Overlap Analysis & Browser Verification:
  - **Overlap@5 Analysis ($N=52$ jobs, 780 slots):** Deduplicated union candidates = 739 (14.21 / job). Multi-system overlap = 41 slots (5.26%). Pairwise overlap@5: BM25 & BGE = 3.46% (0.173 / 5), BM25 & E5 = 1.54% (0.077 / 5), BGE & E5 = 11.54% (0.577 / 5). Directly matches earlier dev-pool pattern (~3–5% lexical-dense overlap). Documented in `data/processed/results/gold_overlap_k5.json` (SHA-256: `71518cc5c3b8d6d7fcc41ec0e98dd9f4e81e1bde8f13d484de35fecde19e4afd`).
  - **Overlap@3 Analysis ($N=52$ jobs, 468 slots):** Deduplicated union candidates = 446 (8.58 / job). Multi-system overlap = 22 slots (4.70%). Pairwise overlap@3: BM25 & BGE = 2.56%, BM25 & E5 = 0.64%, BGE & E5 = 10.90%. Documented in `data/processed/results/gold_overlap_k3.json` (SHA-256: `23e0cbf9fb4db83e3968149c5cdf5a276aedb1c0c96af5fc60ffcd2aea819a74`).
  - **Local Browser Test (Firefox Headless, 0 Network Calls, Independent Flags & Dropdown Reasons):** Executed `labeler_gold.html` (SHA-256: `e4b5ec6919541575d7ffdb556c1f4754ff7d1ffc33e1e8af67130598adeda271`) offline using local Selenium driving `/usr/bin/firefox` (Mozilla Firefox 91.9.0esr). Loaded `annotator_a_tasks.json`, scored 3 pairs across 2 jobs showing both flags can be set independently: Pair 1 (`cannot_judge_job = false`, `is_invalid = false`, score 3, flag_reason = null), Pair 2 (`cannot_judge_job = false`, `is_invalid = true`, score -1, flag_reason = "CV text is empty/broken/unreadable"), Pair 3 (`cannot_judge_job = true`, `is_invalid = false`, score = null, flag_reason = "role outside my expertise", with notes). Verified dropdown reason selector enforcement and mid-job page reload / session resume. Saved sample output: `data/processed/sample_saved_annotations.json` (SHA-256: `970b1d0eb51edf24f05d9469ac1f33e3fe015cc047e49bbd00a63aefc27a9858`).





- **2026-10-01:** Doc consistency fixes (no data, code or frozen files changed): README judge-tuning row now cites 5 tuning jobs (n=45; QWK 0.5517 single-order / 0.6593 order-averaged) instead of "15-job pilot, QWK 0.5284" (0.5284 had no source in logs or docs and was removed); README/status dedup wording now states frozen splits used unigram 0.85 and the 5-gram 0.70 pass was train-side only; status.md Tuning/Splitting/Current Phase lines updated to match decisions.md.
