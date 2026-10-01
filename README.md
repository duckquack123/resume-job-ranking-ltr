# Do We Need LLM Re-Rankers? A Cost-Quality and Fairness Study of Feature-Based LTR for Resume-Job Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## 1. Overview & Research Question

**Central Research Question:**  
*Can a feature-based Learning-to-Rank (LTR) model match the ranking quality of a large language model (LLM) re-ranker for resume-job matching at a fraction of the inference cost, and does it exhibit different fairness and robustness properties?*

### Research Hypotheses
- **H1 (Feature Groups):** The ranking quality gap between feature-based LTR (LambdaMART) and LLM re-rankers shrinks as complementary feature groups (lexical, dense embeddings, cross-encoders, and structured fit signals) are incrementally added.
- **H2 (Ambiguity):** Structured features (e.g., token-length invariants, technical stack overlap) yield the greatest marginal gains on ambiguous queries where lexical and embedding similarities disagree.
- **H3 (Robustness):** Tree-based LTR models demonstrate higher rank stability across random seeds and greater distributional consistency across role families compared to generative LLM re-rankers.
- **H4 (Fairness):** Rankings shift measurably under counterfactual perturbations of protected-attribute proxies; model families differ in counterfactual sensitivity, and simple mitigation constraints trade utility for lower demographic sensitivity.

---

## 2. Current Project Status

| Phase / Milestone | Status | Key Findings & Metrics |
|---|---|---|
| **Phase 1: Data Pipeline & Splitting** | **Complete** | Djinni dataset cleaned; frozen splits built with word-unigram dedup (threshold 0.85). A word-5-gram (0.70) cross-split dedup was then applied to the train side only (frozen dev/test unchanged). 137,893 Train jobs / 168,137 CVs; 27 frozen Dev jobs / 21,019 CVs; 72 frozen Test jobs / 21,020 CVs. |
| **LLM Judge Tuning & Validation** | **Complete & Frozen** | Llama-3.1-8B-Instruct prompt tuned on 5 tuning jobs ($n=45$ pairs; QWK 0.5517 single-order, 0.6593 order-averaged), validated one-time on 5 held-out jobs ($n=41$ pairs; QWK 0.3734 [0.0797, 0.6113], Spearman $r_s = 0.5991$ [0.3245, 0.7938], order-averaged; small, 5-job cluster). Frozen v4 template executed on 4,283 training pairs. |
| **Step 1: LambdaMART Baseline** | **Complete & Frozen** | 15 scale-free features (within-job z-score, ratio-to-top, rank). 5-fold CV NDCG@10 = **0.9221** (vs random **0.8510**). Evaluated on 86 human dev pairs: LambdaMART = 0.8172 vs BGE = 0.8353 vs Random = 0.7593 ($\Delta = +0.0579$, 95% CI `[-0.0140, +0.1305]`). All models, folds, features, and configs frozen and hashed. |
| **Step C: Gold Test Set Preparation** | **IN PROGRESS (Labeling)** | 52 eligible engineering test jobs pooled ($K=3$ per retriever: BM25, BGE, E5; 446 unique pairs). Offline labeling tool (`labeler_gold.html`) verified locally with 0 network calls. Task files partitioned for Annotator A (446 pairs) and Annotator B (115 double-labeled pairs). |
| **Step A: Cross-Encoder Integration** | *Pending* | Small cross-encoder model training and out-of-fold scoring across company-grouped folds. |
| **Step B: LLM Re-Ranker Evaluation** | *Pending* | Zero-shot Qwen2.5-7B-Instruct re-ranking on 86 dev pairs and 52 gold test jobs. |
| **Fairness & Counterfactual Audit** | *Pending* | Counterfactual perturbation analysis on protected-attribute proxies across models. |

---

## 3. Reproduction & Data Sources

This study uses publicly available datasets and does not distribute raw candidate CVs or proprietary job descriptions directly within the repository.

### Data Sources
1. **Djinni English Job Postings:**  
   Hugging Face Hub: [`lang-uk/recruitment-dataset-job-descriptions-english`](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english) (MIT License)
2. **Djinni English Candidate Profiles:**  
   Hugging Face Hub: [`lang-uk/recruitment-dataset-candidate-profiles-english`](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english) (MIT License)
3. **TalentCLEF 2026 (Task A):**  
   Zero-shot transfer evaluation benchmark (unseen test set).

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/username/rjm-ltr.git
cd rjm-ltr

# Create and activate environment
conda create -n rjm-ltr python=3.10 -y
conda activate rjm-ltr

# Install dependencies
pip install torch sentence-transformers lightgbm rank-bm25 datasketch datasets pandas numpy scipy scikit-learn
```

### Reproducing Data Pipeline & Model Training
```bash
# 1. Run data ingestion, language filtering, role-family mapping, and LSH deduplication:
python src/data_pipeline.py

# 2. Run cross-split near-duplicate deduplication (word 5-gram Jaccard threshold 0.70):
bash scripts/slurm/run_dedup.sh

# 3. Extract scale-free features across training and dev pools:
python src/extract_features.py

# 4. Train and evaluate the frozen LambdaMART baseline:
python src/freeze_step1_model.py
```

> **Note on SLURM Cluster Scripts (`scripts/slurm/`):** Batch execution scripts located in `scripts/slurm/` are cluster-specific (configured for our institutional compute cluster, GPU partitions, and local environment paths). Users running on other clusters or workstations should adapt partition flags, GPU resources, working directories, and interpreter paths accordingly.

---

## 4. Model Artifacts & Licensing Notice

- **Code License:** All source code, annotation interfaces, and evaluation scripts in this repository are licensed under the **[MIT License](LICENSE)**.
- **Model Artifacts Notice:** Trained model binary artifacts (`models/*.txt`, checkpoints) are intentionally excluded from the public repository in compliance with the **[Meta Llama 3.1 Community License Agreement](https://llama.meta.com/llama3/license/)** (regarding downstream model distribution and attribution policies) and repository storage best practices.
- **Deterministic Replication:** To ensure exact, reproducible execution without distributing raw weights, frozen parameter configurations and cryptographically verified SHA-256 hashes are recorded in `docs/decisions.md` and `data/processed/frozen_step1_hashes.json`.

---

## 5. Research Ground Rules

All experiments in this repository strictly enforce the pre-registered methodology defined in [`AGENTS.md`](AGENTS.md):
1. **Zero Fabrication:** Never fabricate numbers, citations, or synthetic results. Unknowns are explicitly marked `TODO` or `UNKNOWN`.
2. **Statistical Rigor:** Every reported metric is derived from logged runs and reported with sample size ($n$), 95% confidence intervals (job-clustered / paired bootstrap), and paired comparison tests.
3. **Label Independence:** Supervision labels are derived strictly from human evaluations and independent LLM rubric judgments—never from retrieval scores, lexical overlap, or model features.
4. **Leakage Prevention:** Split assignments are partitioned strictly by job and company; resume pools are disjoint; cross-split near-duplicates are purged (unigram 0.85 for the frozen splits; 5-gram 0.70 audit/train-side removal).
5. **Pre-Gold Freeze:** Hyperparameters, feature schemas, model seeds, and fold definitions are permanently locked and hashed prior to evaluating gold test labels. Gold test labels are evaluated strictly once.
6. **Privacy & Offline Integrity:** No external cloud-based annotation platforms or external web requests are used during annotation; all task data remains local and offline.

---

## 6. What's Still Pending

- [ ] **Human Gold Annotation:** Double-annotation of the 52-job gold test set (446 pairs, 115 double-annotated) and adjudication.
- [ ] **Step A (Cross-Encoder):** Integration of cross-encoder out-of-fold scores as a fourth feature group in LambdaMART.
- [ ] **Step B (LLM Re-Ranker):** Zero-shot Qwen2.5-7B-Instruct re-ranking with digit probabilities and order-swap averaging.
- [ ] **Single-Shot Gold Evaluation:** NDCG@3, NDCG@10, and MRR ($\ge 2$) evaluation across all model variants with company-cluster bootstrap CIs.
- [ ] **Fairness & Counterfactual Audit:** Evaluation of demographic parity and ranking sensitivity under protected-attribute proxy substitutions.

---

## 7. Documentation Pointers

- [`docs/decisions.md`](docs/decisions.md): Comprehensive, dated audit trail of all architectural decisions and cryptographic file hashes.
- [`docs/status.md`](docs/status.md): High-level tracking of active phases, completed milestones, and approved protocols.
- [`docs/step1_evaluation_tables.md`](docs/step1_evaluation_tables.md): Complete evaluation tables, feature shift distributions, ablations, and per-job metrics for Step 1.
- [`docs/candidate_profiles_readme.md`](docs/candidate_profiles_readme.md): Schema and source details for candidate resumes.
- [`docs/job_descriptions_readme.md`](docs/job_descriptions_readme.md): Schema and source details for job postings.
