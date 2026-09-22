# Research Proposal

**Title (working):** *Do We Need LLM Re-Rankers? A Cost–Quality and Fairness Study of Feature-Based Learning-to-Rank for Resume–Job Matching*

**Date:** 2026-09-19 | **Phase 0 Draft**

---

## 1. Research Questions

**Primary RQ:** How does a feature-based LambdaMART re-ranker over heterogeneous signals compare to an LLM-based re-ranker on resume–job matching quality, cost, and fairness sensitivity?

**Sub-questions:**
- RQ1: How does the quality gap between LambdaMART and the LLM re-ranker change as feature diversity increases?
- RQ2: Which feature groups contribute most, and under what query conditions?
- RQ3: Which approach is more stable across random seeds and robust across industries/domains?
- RQ4: How do rankings change under counterfactual edits of protected-attribute proxies, and do model families differ in sensitivity?

---

## 2. Hypotheses (Stated Neutrally)

> All hypotheses are stated as testable claims. We will report the result regardless of which model wins.

- **H1 (Feature diversity effect):** The quality gap between LambdaMART and the LLM re-ranker changes as feature groups are added incrementally (lexical → dense → cross-encoder → structured). *We measure whether this gap shrinks, stays constant, or widens.*
- **H2 (Structured features):** Structured features (skill coverage, experience gap, seniority match, location) have a measurably different impact on queries where embedding similarity alone produces ambiguous rankings versus queries where embeddings are already discriminative. *We measure the per-query nDCG lift stratified by query difficulty.*
- **H3 (Stability and robustness):** LambdaMART and the LLM re-ranker differ in stability across random seeds and in robustness across industry/domain splits. *We measure seed variance (≥3 seeds) and leave-one-industry-out nDCG, reporting whichever model is more stable.*
- **H4 (Fairness sensitivity):** Rankings change measurably under counterfactual edits of protected-attribute proxies (names, gender cues, college tier, graduation year), and model families (feature-based, LLM) differ in their sensitivity. Simple mitigations (masking, feature removal) trade off utility for sensitivity reduction. *We measure sensitivity for all models and report the tradeoff.*

---

## 3. Baselines

| System | Type | Stage |
|---|---|---|
| BM25 (tuned k1, b) | Lexical retrieval | First-stage |
| E5-small-v2 / BGE-small-en-v1.5 | Off-the-shelf dense retrieval | First-stage |
| Fine-tuned bi-encoder (in-batch + hard negatives) | Contrastive dense retrieval | First-stage |
| Cross-encoder (fine-tuned MiniLM) | Neural re-ranker | Second-stage |
| LambdaMART (G1 only) | Feature-based re-ranker (lexical features) | Second-stage |
| LambdaMART (G1+G2) | Feature-based re-ranker (+ dense features) | Second-stage |
| LambdaMART (G1+G2+G3) | Feature-based re-ranker (+ cross-encoder) | Second-stage |
| LambdaMART (G1+G2+G3+G4) | Feature-based re-ranker (all features) | Second-stage |
| LLM re-ranker (pointwise, ~3B) | LLM re-ranker (small) | Second-stage |
| LLM re-ranker (pointwise, ~7B) | LLM re-ranker (large) | Second-stage |
| LLM re-ranker (listwise, ~7B) | LLM re-ranker (if compute allows) | Second-stage |

---

## 4. Metrics

### Quality
- nDCG@{5, 10, 100}
- MRR
- Recall@{10, 100}
- MAP

### Cost
- Latency per query (ms, wall-clock)
- GPU-hours per full evaluation
- Tokens in/out (for LLM re-ranker)

### Fairness Sensitivity
- Mean absolute score change under perturbation
- Mean rank displacement
- Top-k flip rate (fraction of queries where top-k set changes)
- Exposure disparity across perturbation groups

### Statistical
- Paired bootstrap (10,000 resamples) for all key comparisons
- 95% confidence intervals
- Holm–Bonferroni correction for multiple comparisons

---

## 5. Planned Tables and Figures

| # | Type | Content | Hypothesis |
|---|---|---|---|
| T1 | Table | Main comparison: all systems × all quality metrics + latency + cost | H1 |
| T2 | Table | Feature-group ablation: leave-one-out and add-one-at-a-time | H1, H2 |
| T3 | Table | Training-size curve: LambdaMART quality at 10%, 25%, 50%, 100% data | — |
| T4 | Table | Leave-one-industry-out robustness | H3 |
| T5 | Table | Seed variance (mean ± std for each system, ≥3 seeds) | H3 |
| T6 | Table | Fairness sensitivity: score change, rank displacement, flip rate | H4 |
| T7 | Table | Fairness mitigation tradeoff: nDCG@10 vs. sensitivity metric | H4 |
| F1 | Figure | Pareto plot: nDCG@10 vs. latency (log scale) | H1 |
| F2 | Figure | Pareto plot: nDCG@10 vs. cost per 1,000 queries | H1 |
| F3 | Figure | Feature importance (SHAP beeswarm) for LambdaMART | H2 |
| F4 | Figure | Per-query nDCG lift from structured features, stratified by query difficulty | H2 |
| F5 | Figure | Box plot: score distributions under counterfactual perturbations | H4 |

---

## 6. Threats to Validity

### Internal
- **Data leakage:** Mitigated by job-level splits and out-of-fold feature generation. Tested explicitly.
- **Label validity (R12):** Labels must be independent of features. If using weak labels, no labels from skill overlap or LLM family overlap. False negatives from pooled negatives acknowledged.
- **Hyperparameter sensitivity:** Optuna search (≥50 trials) and multi-seed evaluation mitigate this.

### External
- **Synthetic data:** TalentCLEF 2026 data is synthetically generated; results may not transfer to real-world resumes. If fallback is used, semi-synthetic benchmark with weak labels is noisier.
- **Dataset scale:** Shared-task data may be smaller than production systems. Training-size curve helps quantify this risk.
- **Single domain:** If the dataset lacks industry diversity, leave-one-industry-out analysis is limited. We document which industries are represented.

### Construct
- **Fairness as sensitivity:** Our fairness audit measures *sensitivity to perturbations*, not proof of discrimination or fairness. We state this clearly.
- **LLM re-ranker as baseline:** We use zero-shot or few-shot prompting, not a fine-tuned LLM re-ranker (as in ConFit v3). This is intentional (reflects the "plug-and-play" use case) but means the LLM re-ranker is not maximally optimized.

---

## 7. Risks

See the updated risk register in the implementation plan. Key additions per your requirements:

- **R12 (Label validity):** Labels must not be derived from features. No skill-overlap labels if skill overlap is a feature. No LLM-family overlap between label source and LLM re-ranker. False negatives from pooled negatives documented.
- **Annotation workstream:** If fallback data is needed, ~100 jobs × 15–20 candidates, with pilot (~30 jobs), rubric, IAA, running weeks 2–4 in parallel.

---

## 8. Proposed LLM Models

Based on licence and GPU constraints:

| Model | Params | Licence | VRAM (fp16) | Role |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 7.6B | Apache 2.0 | ~15 GB | Larger LLM re-ranker |
| Phi-3.5-mini-instruct | 3.8B | MIT | ~7.6 GB | Smaller LLM re-ranker (preferred over Qwen2.5-3B due to permissive licence) |

> [!NOTE]
> If your GPUs are ≥24 GB (likely, given nodes have 240+ GB RAM and 2 GPUs each): both models fit comfortably. If GPUs are smaller (e.g., 16 GB), we can use 4-bit quantization (GPTQ/AWQ) for the 7B model. Final model selection awaits GPU confirmation.

---

## 9. Schedule with Annotation Workstream

| Week | Main Track | Annotation Track (if fallback needed) |
|---|---|---|
| 1 | Phase 0: Lit review, dataset audit, proposal | — |
| 2 | Phase 1: Data pipeline, loaders, splits, EDA | **Pilot annotation:** rubric draft, ~30 jobs annotated, IAA on pilot |
| 3 | Phase 2: Retrieval baselines, eval harness | **Main annotation:** ~50 more jobs |
| 4 | Phase 2 (cont.) + Phase 3: Feature engineering | **Finalize annotation:** remaining ~20 jobs, IAA report |
| 5 | Phase 4: LambdaMART training, HPO, SHAP | — |
| 6 | Phase 5: LLM re-ranker + Phase 6: Main experiments | — |
| 7 | Phase 6 (cont.) + Phase 7: Fairness audit (stretch) | — |
| 8 | Phase 8: Polish + Phase 9: Paper draft (stretch) | — |

> Core deliverable = Phases 0–6. Phases 7 (fairness) and 9 (paper) are stretch goals.
