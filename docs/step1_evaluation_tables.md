# Step 1 Evaluation & Diagnostic Tables (Revised & Complete)

## 1. Exact Feature Definitions & Split Statistics (Train vs Dev)
All raw-scale features have been replaced by within-job scale-free versions (z-score, ratio to top candidate, and rank). Within each job pool:
- `_zscore`: $(x - \mu_{\text{job}}) / (\sigma_{\text{job}} + 10^{-6})$ (centered at 0.0, unit variance)
- `_ratio_top`: $x / (\max_{\text{job}}(x) + 10^{-6})$ (bounded in $[0, 1]$, top candidate = 1.0)
- `_rank`: rank within job pool ($1 = \text{top candidate}$)

### Final Model Feature Set (15 Exact Features)
*Note on Z-Scores:* Within-job z-scores have mean 0.0 and std 1.0 by mathematical definition within each query group; claiming that z-scores "match" across splits is true by construction and conveys no empirical information about feature distributions. The true substantive scale-invariant ranking features are within-job **ratios-to-top** and **ranks**, whose empirical distributions across splits are reported below.

| Feature Name | Feature Group | Train Mean $\pm$ Std | Train Median [IQR] | Dev Mean $\pm$ Std | Dev Median [IQR] |
|---|---|---|---|---|---|
| `bm25_ratio_top` | Lexical | 0.659 $\pm$ 0.249 | 0.610 [0.458–0.936] | 0.771 $\pm$ 0.213 | 0.771 [0.638–0.989] |
| `bm25_rank` | Lexical | 4.887 $\pm$ 2.533 | 5.000 [3.000–7.000] | 4.814 $\pm$ 2.504 | 5.000 [3.000–7.000] |
| `bge_ratio_top` | Dense | 0.929 $\pm$ 0.065 | 0.938 [0.887–0.990] | 0.914 $\pm$ 0.082 | 0.927 [0.859–0.989] |
| `bge_rank` | Dense | 4.887 $\pm$ 2.533 | 5.000 [3.000–7.000] | 4.814 $\pm$ 2.504 | 5.000 [3.000–7.000] |
| `e5_ratio_top` | Dense | 0.977 $\pm$ 0.021 | 0.979 [0.962–0.997] | 0.975 $\pm$ 0.022 | 0.982 [0.958–0.992] |
| `e5_rank` | Dense | 4.887 $\pm$ 2.533 | 5.000 [3.000–7.000] | 4.814 $\pm$ 2.504 | 5.000 [3.000–7.000] |
| `cv_token_len_ratio_top` | Structured | 0.554 $\pm$ 0.275 | 0.507 [0.321–0.790] | 0.506 $\pm$ 0.284 | 0.461 [0.286–0.717] |
| `cv_token_len_rank` | Structured | 4.879 $\pm$ 2.531 | 5.000 [3.000–7.000] | 4.802 $\pm$ 2.510 | 5.000 [3.000–7.000] |
| `tech_overlap_ratio_top` | Structured | 0.507 $\pm$ 0.374 | 0.500 [0.143–0.833] | 0.447 $\pm$ 0.367 | 0.500 [0.000–0.714] |
| `tech_overlap_rank` | Structured | 3.747 $\pm$ 2.493 | 3.000 [1.000–6.000] | 3.744 $\pm$ 2.450 | 3.000 [1.000–6.000] |
| `bm25_zscore` | Lexical | 0.000 $\pm$ 1.000 | 0.000 [True by construction] | 0.000 $\pm$ 1.006 | 0.000 [True by construction] |
| `bge_zscore` | Dense | 0.000 $\pm$ 1.000 | 0.000 [True by construction] | 0.000 $\pm$ 1.006 | 0.000 [True by construction] |
| `e5_zscore` | Dense | 0.000 $\pm$ 1.000 | 0.000 [True by construction] | 0.000 $\pm$ 1.006 | 0.000 [True by construction] |
| `cv_token_len_zscore` | Structured | 0.000 $\pm$ 1.000 | 0.000 [True by construction] | 0.000 $\pm$ 1.006 | 0.000 [True by construction] |
| `tech_overlap_zscore` | Structured | 0.000 $\pm$ 0.969 | 0.000 [True by construction] | 0.000 $\pm$ 0.952 | 0.000 [True by construction] |

*Corpus Shift Note:* Raw BM25 scores differed drastically between Train ($N=168,137$, mean $42.51 \pm 20.09$) and Dev ($N=21,019$, mean $190.51 \pm 99.89$). Within-job ratio-to-top and rank transformations insulate the ranker from cross-split scale mismatch, though slight distribution differences remain from differing query lengths and candidate pool compositions.

---

## 2. Specific Feature Ablations on CV Length & Tech Overlap
Models trained on judge labels with company-grouped 5-fold CV, and evaluated on dev human labels.

| Configuration | Features Included | 5-Fold CV Out-of-Fold (vs Judge) | Tuning (45 pairs) (vs Human) | Held-Out (41 pairs) (vs Human) | All Dev (86 pairs) (vs Human) |
|---|---|---|---|---|---|
| **Full Model** | All 15 features | **0.9221** | 0.7289 | 0.9056 | 0.8172 |
| **Drop CV Length** | 12 features (Drop `cv_token_len_*`) | 0.9160 | **0.8260** | 0.8784 | **0.8522** |
| **Drop Tech Overlap** | 12 features (Drop `tech_overlap_*`) | 0.9067 | 0.7509 | **0.9345** | 0.8427 |
| **Drop Both (Lex+Dense)** | 9 features (Lexical + Dense only) | 0.9022 | 0.8060 | 0.8714 | 0.8387 |
| **Tech Overlap Alone** | 3 features (`tech_overlap_*`) | 0.9045 | 0.7514 | 0.8985 | 0.8250 |
| **CV Length Alone** | 3 features (`cv_token_len_*`) | 0.8598 | 0.6722 | 0.8497 | 0.7609 |
| *Random Baseline* | *Expected uniform permutation* | *0.8510* | *0.7200* | *0.7986* | *0.7593* |

### CV Length Spearman Correlations
- **With Judge Grades on Train ($n = 4,283$ pairs):** $r_s = \mathbf{0.0358}$ ($p = 0.019$)
- **With Human Grades on Dev ($n = 86$ pairs):** $r_s = \mathbf{0.0671}$ ($p = 0.539$)
*Finding:* Candidate CV length has near-zero correlation with both judge grades and human grades.

---

## 3. Paired Bootstrap over 10 Jobs (LambdaMART vs BGE vs Random)
Per-job NDCG@10 scores on the 86 human annotations across all 10 dev jobs.

### Per-Job Evaluation Table ($n = 10$ jobs, 86 pairs)
| Job ID | Split | Candidates ($m$) | LambdaMART (Full) | BGE-small | Random Baseline | $\Delta$ (LM - BGE) | $\Delta$ (LM - Rand) |
|---|---|---|---|---|---|---|---|
| `105723_job` | Tuning | 9 | 0.8863 | 0.8594 | 0.7074 | +0.0269 | +0.1789 |
| `110001_job` | Tuning | 9 | 0.8706 | 0.9147 | 0.9127 | -0.0440 | -0.0420 |
| `26348_job` | Tuning | 9 | 0.4486 | 0.4575 | 0.5797 | -0.0089 | -0.1311 |
| `53525_job` | Tuning | 9 | 0.7435 | 0.8535 | 0.6753 | -0.1100 | +0.0682 |
| `75798_job` | Tuning | 9 | 0.6954 | 0.8948 | 0.7251 | -0.1995 | -0.0297 |
| **Tuning Mean** | | | **0.7289** | **0.7960** | **0.7200** | **-0.0671** | **+0.0089** |
| `129445_job` | Held-Out | 8 | 0.8033 | 0.7812 | 0.8209 | +0.0221 | -0.0176 |
| `22047_job` | Held-Out | 8 | 0.9822 | 0.8084 | 0.8474 | +0.1738 | +0.1348 |
| `50223_job` | Held-Out | 9 | 0.9935 | 0.9635 | 0.7251 | +0.0300 | +0.2684 |
| `64381_job` | Held-Out | 8 | 0.9081 | 0.9252 | 0.7358 | -0.0171 | +0.1723 |
| `70278_job` | Held-Out | 8 | 0.8409 | 0.8948 | 0.8641 | -0.0539 | -0.0232 |
| **Held-Out Mean** | | | **0.9056** | **0.8746** | **0.7986** | **+0.0310** | **+0.1069** |
| **Overall Mean** | | | **0.8172** | **0.8353** | **0.7593** | **-0.0181** | **+0.0579** |

### Paired Cluster Bootstrap Intervals (1,000 resamples over 10 jobs)
- **LambdaMART vs BGE:** Mean diff = **-0.0181**, 95% CI: `[-0.0784, +0.0377]`
  - *Plain Statement:* **BGE alone scores above LambdaMART overall** (0.8353 vs 0.8172). While LambdaMART outperforms BGE on held-out jobs (0.9056 vs 0.8746), it underperforms BGE on tuning jobs (0.7289 vs 0.7960).
  - *Plain Statement on Tuning Pairs:* On the 45 tuning pairs, some LambdaMART variants score **below random** (e.g., CV Length Alone 0.6722 vs Random 0.7200, and in raw-feature model 0.7062 vs 0.7200). The full scale-free model achieves 0.7289 vs Random 0.7200, performing essentially at baseline.
- **LambdaMART vs Random:** Mean diff = **+0.0579**, 95% CI: `[-0.0140, +0.1305]`
- **BGE vs Random:** Mean diff = **+0.0760**, 95% CI: `[+0.0068, +0.1485]`
- **5-Fold CV NDCG@10 (Full Model vs Judge):** Mean = **0.9221**, 95% Job-Cluster CI: `[0.9150, 0.9291]` ($n = 490$ jobs).

---

## 4. Provenance Accounting & Multi-System Retrieval
The candidate pool was constructed by taking top-$K=3$ candidates per system (BM25, BGE, E5) across 10 dev jobs ($10 \times 3 = 30$ slots per system = 90 slots).
Due to multi-system retrieval (4 candidates retrieved by $>1$ system), the deduplicated union pool contains **86 unique candidate pairs**.

### Marginal Retrieval per System
*Candidates retrieved by multiple systems are counted in EACH retrieving system's pool ($N=30$ per system).*

| Retrieval System | Candidates ($N$) | Mean Human Grade | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Relevant Rate ($\ge 2$) | Top Grade Rate ($= 3$) |
|---|---|---|---|---|---|---|---|---|
| **BGE-small** | 30 | **1.93** | 3 | 8 | 7 | **12** | **63.3%** (19/30) | **40.0%** (12/30) |
| **E5-small** | 30 | 1.70 | 8 | 3 | 9 | 10 | **63.3%** (19/30) | **33.3%** (10/30) |
| **BM25** | 30 | 1.07 | 14 | 5 | 6 | 5 | **36.7%** (11/30) | **16.7%** (5/30) |

*Explanation of 63.3% / 36.7% Figures:*
- For BGE: 7 Grade 2 + 12 Grade 3 = 19 relevant candidates out of 30 $\implies 19/30 = \mathbf{63.33\%}$.
- For E5: 9 Grade 2 + 10 Grade 3 = 19 relevant candidates out of 30 $\implies 19/30 = \mathbf{63.33\%}$.
- For BM25: 6 Grade 2 + 5 Grade 3 = 11 relevant candidates out of 30 $\implies 11/30 = \mathbf{36.67\%}$.

### Mutually Exclusive Partition of 86 Dev Pairs
| Partition | Count ($N$) | Percentage | Mean Human Grade | $\% \ge 2$ | $\%=3$ |
|---|---|---|---|---|---|
| **Exclusively BGE** | 27 | 31.4% | 1.89 | 63.0% (17/27) | 37.0% (10/27) |
| **Exclusively E5** | 26 | 30.2% | 1.62 | 61.5% (16/26) | 30.8% (8/26) |
| **Exclusively BM25** | 29 | 33.7% | 1.03 | 34.5% (10/29) | 17.2% (5/29) |
| **Retrieved by Multiple** | 4 | 4.7% | **2.25** | **75.0%** (3/4) | **50.0%** (2/4) |
| **Total Union Pool** | **86** | 100.0% | 1.50 | 53.5% (46/86) | 29.1% (25/86) |

---

## 5. Label Distribution Comparison & Quantile Sensitivity Variant

### Label Distribution Comparison
| Grade | Description | Judge on Train ($N=4,283$) | Human on Dev ($N=86$) | Discrepancy Analysis |
|---|---|---|---|---|
| **0** | Irrelevant | 316 (**7.38%**) | 25 (**29.07%**) | Judge is far more lenient; rarely assigns Grade 0 |
| **1** | Marginally Relevant | 898 (**20.97%**) | 15 (**17.44%**) | Roughly comparable |
| **2** | Somewhat Relevant | 2,374 (**55.43%**) | 21 (**24.42%**) | Judge clusters mass heavily into Grade 2 mode |
| **3** | Highly Relevant | 695 (**16.23%**) | 25 (**29.07%**) | Human annotator assigns Grade 3 nearly 2x more often |

*Takeaway:* The judge distribution is unimodal around Grade 2, whereas the human distribution is U-shaped (heavy tails at 0 and 3).

### Sensitivity Variant: Quantile-Binned Expected Scores (4 Bins)
*Trained on 4 equal quantile bins of continuous judge expected scores. Primary target remains rounded average grades.*

| Metric | Primary Model (Rounded Grades 0–3) | Sensitivity Variant (Quantile Bins 0–3) |
|---|---|---|
| **5-Fold CV NDCG@10 (Out-of-Fold)** | **0.9221** (Random: 0.8510) | **0.8578** (Random: 0.7636) |
| **Tuning (45 pairs) NDCG@10** | 0.7289 | **0.7467** |
| **Held-Out (41 pairs) NDCG@10** | 0.9056 | **0.9533** |
| **All Dev (86 pairs) NDCG@10** | 0.8172 | **0.8500** |

*Sensitivity Finding:* Quantile binning flattens the judge mode and slightly improves transfer to human NDCG@10 on dev (0.8500 vs 0.8172). However, primary target remains rounded average grades per specification; quantile binning is recorded strictly as a sensitivity check.

---

## 6. Folds, Hyperparameters, and Execution Runtime Verification
- **Folds:** `train_500_cv_folds.json` (frozen, company-grouped 5 folds) was RESTRICTED to 490 jobs by filtering out the 10 excluded job IDs (`excluded_train_job_ids_v2.txt`). Folds were NOT regenerated or reshuffled.
- **Tree Count & Early Stopping:** `n_estimators=100`, with early stopping after 15 rounds of no improvement on out-of-fold validation NDCG@10. In 5-fold CV, validation peak was reached at an average of 38 trees across folds. The final production booster `models/lambdamart_step1_frozen.txt` was trained for 100 trees on all 4,283 pairs.
- **Hyperparameter Selection (Judge CV Only):** Hyperparameters (`learning_rate=0.05`, `num_leaves=15`, `min_child_samples=10`) were chosen exclusively via 5-fold company-grouped cross-validation on the 490 training jobs' judge labels (evaluating CV NDCG@10 across a grid of learning rates {0.01, 0.05, 0.1}, num_leaves {7, 15, 31}, min_child_samples {5, 10, 20}), **NEVER on human labels**. Human labels and dev sets were never touched during hyperparameter tuning.
- **Judge Run Runtime:** 24 minutes 43 seconds (1,483 seconds) for 4,283 pairs (8,566 prompts) on single A30 GPU under SLURM (**2.89 pairs/second**).

---

## 7. Retrieval Performance Interpretation (Pool Construction Caveat)
BM25 (0.7337) and E5 (0.7524) scoring at or below random baseline (0.7593) on the 86 human pairs reflects candidate pool construction, not general retrieval failure. Because all candidates were pooled from the top-3 of BM25, BGE, and E5, every candidate in the evaluation pool is already in the top 0.01% of the corpus for at least one system. Re-ranking among pre-filtered, highly similar candidates diminishes lexical and dense discriminative power relative to an unconstrained corpus search.

---

## 8. Step C Gold Test Set Pooling & Task Export Statistics
- **Eligible Test Jobs ($N=52$):** 52 jobs evaluated from `jobs_test.parquet` using `is_eligible` blocklist + `eligibility_overrides.csv`. Includes `134642_job` ("TDM with focus on QA\QC (VR-61299)" at Luxoft), confirmed as technical QA Automation / Test Delivery Management.
- **Test CV Pool ($N=21,020$):** `cvs_test.parquet` with `excluded_cv_ids.txt` applied (`114596_cv` excluded).
- **Pooling System ($K=3$):** BM25 + BGE-small-en-v1.5 + E5-small-v2 (total retrieval slots = $52 \times 9 = 468$).
- **Within-Pool Candidate Dedup (5-gram Jaccard $\ge 0.70$):** 0 near-duplicates found within any job's candidate pool.
- **Unique Candidate Pairs ($N=446$):** 22 candidates retrieved by $\ge 2$ systems. Average candidate pool = 8.58 candidates/job.
- **Annotator B Subsampling (13 Jobs, Seed 42):** Stratified by Role Family (Javascript: 3, QA Automation: 3, Other: 2, Java: 1, DevOps: 1, Ruby: 1, Python: 1, PHP: 1). Hashed before labeling: `data/processed/annotator_b_job_ids.txt` (SHA-256: `229cf6abbb9a88f4ce7bacc278a484a719b567cb20fd585a80760e33c53ab3c6`).
- **Task Exports:**
  - **Annotator A (52 jobs, 446 pairs):** `data/processed/annotator_a_tasks.json` (SHA-256: `c1dd961afbc70e04b25a9ff8949b97a759cd57ccacfcd810048f7af853cc228b`).
  - **Annotator B (13 jobs, 115 pairs):** `data/processed/annotator_b_tasks.json` (SHA-256: `28fc6a42bf9d180026e8961706fb80d7d20308e539d51a0ca6b831154129d8c3`).
  - **Provenance Mapping:** `data/processed/gold_provenance.json` (SHA-256: `fcf6544821d3f1b5d9d947e1f45bbccdc11c27b2a8e7e3a11295d866f8cfb69f`).
- **Annotation Interface:** `data/processed/labeler_gold.html` (offline single-page, 3 Strong / 2 Good / 1 Weak / 0 Not relevant / Invalid, on-screen rubric/ignore rules, job notes, cannot-judge flag, timestamps, autosave/resume, 0 network calls).
- **Integrity Check:** `dev_hand_labeling_tasks.json` verified 100% UNTOUCHED (SHA-256: `2e6a5630e7b46e0e46257e58f7285d45cd5970497a67afd92e2800d14b449190`).

---

## 9. Gold Test Set Overlap@5 & Overlap@3 Analysis (52 Jobs, 21,020 Clean Test CVs)
Evaluated across all 52 eligible test jobs pooled from `cvs_test.parquet` (excluding `114596_cv`).

### Overlap Statistics Table
| Metric / System Pair | Overlap@5 (Gold 52 Jobs) | Overlap@3 (Gold 52 Jobs) | Dev Pool Pattern (Earlier Benchmark) |
|---|---|---|---|
| **Total Retrieval Slots** | 780 ($52 \times 15$) | 468 ($52 \times 9$) | 150 ($10 \times 15$) |
| **Deduplicated Union Candidates** | 739 (14.21 / job) | 446 (8.58 / job) | ~14.1 / job |
| **Multi-System Overlap Slots** | 41 (**5.26%**) | 22 (**4.70%**) | ~5.0% |
| **BM25 & BGE-small Overlap** | 0.173 / 5 (**3.46%**) | 0.077 / 3 (**2.56%**) | ~4.0% |
| **BM25 & E5-small Overlap** | 0.077 / 5 (**1.54%**) | 0.019 / 3 (**0.64%**) | ~2.0% |
| **BGE-small & E5-small Overlap** | 0.577 / 5 (**11.54%**) | 0.327 / 3 (**10.90%**) | ~12.0% |

*Interpretation:* The gold pool overlap@5 directly replicates the dev pool pattern (~3–5% lexical-dense overlap, ~11–12% dense-dense overlap). There is no near-total overlap; retriever diversity is high.
- Hashed Overlap Files:
  - `data/processed/results/gold_overlap_k5.json` (SHA-256: `71518cc5c3b8d6d7fcc41ec0e98dd9f4e81e1bde8f13d484de35fecde19e4afd`)
  - `data/processed/results/gold_overlap_k3.json` (SHA-256: `23e0cbf9fb4db83e3968149c5cdf5a276aedb1c0c96af5fc60ffcd2aea819a74`)

---

## 10. Labeler Browser Execution Verification (0 Network Calls, Independent Flags & Dropdown Reasons)
Tested locally via headless Firefox (`/usr/bin/firefox`, Mozilla Firefox 91.9.0esr) with 0 external network requests:
- **Task Loaded:** `annotator_a_tasks.json` (52 jobs, 446 candidate pairs).
- **Navigation & Boundary:** Persistent 52-job navigation sidebar on left; clear job header and rubric on center-left; candidate CV and evaluation panel on right.
- **Independent Flags & Required Dropdown Reasons:**
  - `cannot_judge_job` and `is_invalid` operate independently in UI and saved schema.
  - Required reason dropdown (`"role outside my expertise"` vs `"CV text is empty/broken/unreadable"`) enforced when either flag is set.
- **Pairs Evaluated (3 sample pairs across 2 jobs):**
  1. `101411_job_52004_cv` (Job 1: `101411_job`) $\implies$ Score: **3**, `cannot_judge_job = false`, `is_invalid = false`, `flag_reason = null`.
  2. `101411_job_140379_cv` (Job 1: `101411_job`) $\implies$ Score: **-1**, `cannot_judge_job = false`, `is_invalid = true`, `flag_reason = "CV text is empty/broken/unreadable"`.
  3. `101499_job_158168_cv` (Job 2: `101499_job`) $\implies$ Score: **null**, `cannot_judge_job = true`, `is_invalid = false`, `flag_reason = "role outside my expertise"`, Note entered (`"Domain requires specialized regulatory compliance knowledge outside evaluator scope."`).
- **Saved Output:** `data/processed/sample_saved_annotations.json` (SHA-256: `970b1d0eb51edf24f05d9469ac1f33e3fe015cc047e49bbd00a63aefc27a9858`).
- **Interface File:** `data/processed/labeler_gold.html` (SHA-256: `e4b5ec6919541575d7ffdb556c1f4754ff7d1ffc33e1e8af67130598adeda271`).
- **Reload & Resume Verification:** Page refreshed at Job 2. Restored `cannot_judge_job` checkbox as `True`, restored job reason dropdown as `"role outside my expertise"`, preserved audit notes, retained prior scores, and resumed cleanly.

