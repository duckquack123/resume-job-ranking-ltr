# Related Work

> **Status:** Phase 0 draft. All summaries are based on web search results, abstracts, and publicly available information. Papers marked `VERIFIED` were confirmed via their arXiv, ACL Anthology, or official repository pages. Papers marked `UNVERIFIED` could not be fully read; claims are based on secondary sources.

---

## 1. Person-Job Fit: The ConFit Family

### 1.1 ConFit (Yu et al., 2024) — `VERIFIED` via arXiv:2401.16349
- **Problem:** Label sparsity in resume–job matching: each job seeker applies to only a tiny fraction of available jobs, yielding very few positive (job, resume) pairs.
- **Method:** Contrastive fine-tuning of a bi-encoder (E5 / Jina-v2 backbone). Data augmentation by paraphrasing resume/job sections to increase the effective batch. In-batch negatives scale from B pairs to O(B²) training signals. Standard FAISS inner-product retrieval at inference.
- **Data:** AliYun/Tianchi Person-Job Fit 2019 (Chinese) and IntelliPro (proprietary, English + Chinese).
- **Metrics:** nDCG@10, Recall@10/100. Reported ~19% absolute improvement in nDCG@10 over BM25 and OpenAI text-ada-002 on the job-ranking direction.
- **Baselines:** BM25, text-ada-002, text-embedding-003, DPGNN, MV-CoN.
- **Limitations:** Evaluated on proprietary and competition data that are no longer publicly available. Bi-encoder only; no structured features or re-ranking stage.
- **How our study differs:** We add a re-ranking stage (LambdaMART) over heterogeneous features on top of a bi-encoder retriever. We compare this to an LLM re-ranker rather than only an embedding baseline. We also add a fairness audit, which ConFit does not address.

### 1.2 ConFit v2 (Yu et al., 2025) — `VERIFIED` via arXiv / ACL Anthology
- **Problem:** Same sparsity problem, with additional focus on generating better training signals.
- **Method:** Two additions to ConFit: (1) **HYRE** — use an LLM to generate a "hypothetical reference resume" per job post, creating an additional augmented anchor; (2) **RUM** — runner-up hard-negative mining from unlabeled pairs.
- **Data:** Same AliYun + IntelliPro datasets.
- **Metrics:** +13.8% absolute Recall, +17.5% absolute nDCG over ConFit v1 and text-embedding-003.
- **Limitations:** Still evaluated only on datasets that are not publicly accessible. HYRE depends on an LLM for data augmentation, adding a hidden cost.
- **How our study differs:** We study the LLM as a *re-ranker* (at inference time), not just as a training-time data augmenter. We use publicly accessible data.

### 1.3 ConFit v3 (Yu et al., 2026) — `VERIFIED` via arXiv (May 2026)
- **Problem:** Embedding-based retrieval lacks controllability and explainability. Shift from retrieval to LLM-based re-ranking.
- **Method:** Systematic analysis of the LLM re-ranker training pipeline: multi-pass re-ranking, listwise RL objectives (GRPO), noisy sample removal, SFT distillation from a stronger teacher LLM before RL. Uses Qwen3-8B and Qwen3-32B.
- **Data:** AliYun + IntelliPro. Claims to outperform GPT-5 and Claude Opus-4.5.
- **Metrics:** nDCG@10, Recall@10/100.
- **Limitations:** Proprietary data; large LLMs (32B) require significant compute; comparison to GPT-5/Claude is on their specific data.
- **How our study differs:** We compare a *feature-based* re-ranker (LambdaMART) to an LLM re-ranker rather than comparing LLM families. We study cost–quality tradeoffs explicitly. We use open data and open models.

---

## 2. Learning-to-Rank Foundations

### 2.1 From RankNet to LambdaRank to LambdaMART (Burges, 2010) — `VERIFIED` via MSR-TR-2010-82
- **Problem:** Ranking documents by relevance in web search.
- **Method:** Progression from neural pairwise ranking (RankNet, 2005) → gradient-defined listwise ranking (LambdaRank, 2007) → gradient-boosted trees (LambdaMART, 2010). LambdaMART applies "lambda" gradients (which implicitly optimize NDCG) to GBDT (MART).
- **Data:** LETOR, web search logs.
- **Key insight:** LambdaMART avoids needing a differentiable surrogate for NDCG by directly defining gradients that push relevant documents up. This is the foundation for LightGBM's `lambdarank` objective.
- **How our study uses this:** LambdaMART via LightGBM's `LGBMRanker` is our primary LTR model.

### 2.2 Feature-Based LTR for Recruitment — `UNVERIFIED` (no single canonical paper; based on industry reports and theses)
- **Problem:** Re-ranking candidates for job openings using hand-crafted features.
- **Method:** Two-stage retrieve-then-rank pipelines in production (e.g., LinkedIn's talent search). Features include text similarity, skill overlap, experience match, location proximity, seniority alignment, etc. LambdaMART or similar GBDT rankers are standard.
- **Data:** Typically proprietary (LinkedIn internal data, Indeed, etc.).
- **Key references:** "Learning to Retrieve for Job Matching" (LinkedIn), "Applying machine learning to a job-candidate matching problem" (Leiden thesis). These describe the general approach but use proprietary data and do not publish reproducible benchmarks.
- **Limitations:** No public, reproducible study with feature-group ablation and comparison to LLM re-rankers.
- **How our study differs:** We provide a public, reproducible ablation of feature groups (lexical, dense, cross-encoder, structured) and compare against an LLM baseline, filling this gap.

---

## 3. LLM-Based Re-Ranking

### 3.1 RankGPT (Sun et al., 2023) — `VERIFIED` via arXiv
- **Problem:** Zero-shot document re-ranking using LLMs.
- **Method:** Listwise prompting: the LLM receives a query and a set of passages, then outputs a ranked permutation. Sliding-window strategy for long lists: rank overlapping windows back-to-front, merge results.
- **Data:** TREC-DL, BEIR, NovelEval.
- **Metrics:** nDCG@10. Competitive with or better than supervised cross-encoders in zero-shot settings.
- **Limitations:** High latency (autoregressive generation of permutations). Context-window limits. Expensive for large candidate sets.
- **How our study uses this:** RankGPT-style listwise prompting is one of our LLM re-ranking strategies (if compute allows). We also implement pointwise scoring as a simpler alternative.

### 3.2 Pointwise LLM Re-Ranking — `VERIFIED` (various papers, 2023–2024)
- **Method:** Score each (query, document) pair independently with an LLM (e.g., probability of a "relevant" token). Parallelizable, simpler than listwise.
- **Limitation:** Does not leverage inter-document comparisons; behaves like a neural cross-encoder.
- **How our study uses this:** This is our primary LLM re-ranking strategy (more compute-efficient than listwise).

### 3.3 Efficient LLM Re-Ranking (FIRST, Rank-R1, ICR) — `UNVERIFIED` (based on survey summaries)
- **FIRST:** Uses output logits of the first generated token to rank, reducing latency ~50%.
- **Rank-R1:** Reasoning-intensive re-ranker with chain-of-thought and RL training.
- **ICR:** In-context re-ranking via attention pattern changes, avoiding autoregressive generation.
- **How our study uses this:** We note these as concurrent work but do not implement them; our focus is on the cost–quality comparison between feature-based and standard LLM prompting approaches.

---

## 4. TalentCLEF Shared Tasks

### 4.1 TalentCLEF 2025 — `VERIFIED` via official website
- **Tasks:** (A) Multilingual Job Title Matching, (B) Job Title-Based Skill Prediction.
- **Setup:** Query = job title, Corpus = knowledge base of job titles or skills. **This is NOT a resume–job matching task.** It operates on job titles/skills, not full resume documents.
- **Evaluation:** TREC-format submissions, evaluated by MAP.
- **Relevance to our work:** Not directly usable — the task structure does not match our "job = query, resumes = candidates" setup.

### 4.2 TalentCLEF 2026 Task A: Contextualized Job-Person Matching — `VERIFIED` via Zenodo (DOI: 10.5281/zenodo.19652670) and official website
- **Tasks:** (A) Given a job description, rank candidate résumés by suitability. *This matches our setup exactly.* (B) Job-Skill matching (not relevant).
- **Data:** Synthetically generated from structured resources derived from real job descriptions and résumés. No personal information exposed. English and Spanish.
- **Format:** `queries/` (job descriptions), `corpus_elements/` (candidate résumés), `qrels.tsv` (relevance judgments). Standard TREC evaluation format.
- **Evaluation:** MAP (primary), MRR, P@1/5/10.
- **Licence:** Publicly available on Zenodo. Competition registration closed April 2026, but datasets remain accessible for research.
- **Limitations:** Synthetically generated data; may not reflect real-world resume diversity. Relevance judgments may be binary (needs verification).
- **Relevance to our work:** **This is our primary dataset candidate.** It is public, has the right task structure, and comes with relevance judgments. The synthetic nature is a limitation we must document.

---

## 5. Unbiased Learning to Rank

### 5.1 Unbiased LTR with Biased Feedback (Joachims et al., 2017) — `VERIFIED`
- **Problem:** Click data for LTR is biased by position (higher-ranked items get more clicks regardless of relevance).
- **Method:** Inverse Propensity Weighting (IPW): weight each click by the inverse of its examination probability to de-bias the loss function. Propensity-Weighted Ranking SVM.
- **Relevance to our work:** Our setting uses editorial/synthetic labels, not click data, so position bias is not directly applicable. However, we note that pooled negative sampling introduces a form of selection bias (negatives are only drawn from retrieved candidates, not the full corpus), which relates to the broader ULTR literature.

### 5.2 Doubly-Robust Estimators and DLA — `UNVERIFIED` (based on survey summaries)
- Advanced methods combining IPW with outcome modeling (doubly-robust) or jointly learning propensity and ranking models (DLA).
- **Relevance:** Not directly used in our study but relevant background for the unbiased LTR section of the paper.

---

## 6. Fairness in Algorithmic Hiring

### 6.1 LLM Bias in Resume Screening (Wilson & Caliskan, 2024) — `UNVERIFIED` (based on search summaries)
- **Problem:** LLM-based screening systems replicate or amplify race and gender biases.
- **Method:** Sensitivity testing: inject identity-signaling features (names, pronouns) into otherwise identical resumes. Measure score/rank changes.
- **Findings:** LLMs show sensitivity to demographic cues even when resume content is otherwise identical.
- **How our study uses this:** Our fairness audit (Phase 7) follows a similar counterfactual perturbation methodology.

### 6.2 Counterfactual Audit Methodologies (2024–2026) — `UNVERIFIED`
- **Trend:** Scalable auditing using LLM-generated or template-based counterfactual resumes with controlled demographic variations.
- **Metrics:** Score deltas, rank displacement, top-k flip rate, exposure disparity across perturbation groups.
- **Frameworks:** "AI Bias Firewall" — per-decision bias scores connecting technical score shifts to legal standards of disparate impact.
- **How our study uses this:** We adopt the counterfactual perturbation approach (name swap, college swap, etc.) and the multi-metric evaluation framework.

### 6.3 Demographic Parity vs. Merit-Aware Fairness — `UNVERIFIED`
- **Tension:** Equal selection rates (demographic parity) may conflict with merit-based ranking. Practical frameworks increasingly use per-decision auditing rather than group-level statistical tests.
- **How our study uses this:** We frame our fairness audit as *sensitivity measurement*, not proof of fairness. We report utility–sensitivity tradeoffs and document limitations.

---

## 7. Skill Taxonomies

### 7.1 ESCO (European Skills, Competences, Qualifications and Occupations) — `VERIFIED`
- **What:** EU-maintained multilingual taxonomy of skills, competences, and occupations. Open access, available as CSV/JSON/RDF and via API.
- **Licence:** Free for all use (European Commission).
- **Relevance:** We plan to use ESCO for skill normalization in structured features (G4). Download the CSV/JSON version for offline matching.

---

## Summary Table

| Paper / Resource | Problem | Method | Data | Our Relationship |
|---|---|---|---|---|
| ConFit (2024) | PJF label sparsity | Contrastive bi-encoder | AliYun, IntelliPro (private) | We add re-ranking + ablation |
| ConFit v2 (2025) | PJF hard negatives | HYRE + RUM | Same private data | We compare LLM as re-ranker, not augmenter |
| ConFit v3 (2026) | LLM re-ranking for PJF | Listwise RL, Qwen3 | Same private data | We compare feature-based vs. LLM re-ranker |
| Burges (2010) | Ranking foundations | RankNet → LambdaMART | LETOR | Our core ranking algorithm |
| RankGPT (Sun, 2023) | Zero-shot re-ranking | Listwise LLM prompting | TREC-DL, BEIR | Our LLM re-ranking baseline |
| TalentCLEF 2026 Task A | Job-Person matching | Shared task | Synthetic (Zenodo) | **Primary dataset candidate** |
| Joachims (2017) | Position bias in LTR | IPW | Click logs | Background for bias discussion |
| Wilson & Caliskan (2024) | LLM bias in hiring | Counterfactual audit | Synthetic resumes | Methodological inspiration for Phase 7 |
| ESCO | Skill taxonomy | EU classification | Open access | Skill normalization for G4 features |
