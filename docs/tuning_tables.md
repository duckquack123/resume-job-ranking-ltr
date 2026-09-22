# Prompt Tuning & Tuning-Set Evaluation Tables

## 1. Prompt Versions Tuning Comparison (N = 45 Tuning Pairs)
*Rough check on n = 45 tuning pairs.*

| Version | Description | Template SHA-256 | QWK | Spearman | Exact Agreement | Judge vs CV Len ($r$) |
|---|---|---|---|---|---|---|
| **v1_baseline** | Original generic rubric | `cec806906ebf6b25fec267f7e2fc969667b96ccf704481c68290976ac2540f5a` | 0.4375 | 0.5732 | 33.3% (15/45) | -0.1179 |
| **v2_tech_stack** | Explicit core vs adjacent stack | `373140e18a860632b6bda3a8d3f621b42c95ab821d8f56d19d38f3ba3073f1a7` | 0.4020 | 0.5261 | 33.3% (15/45) | -0.1609 |
| **v3_competency** | Foundational tech & frameworks | `aff2f962b627662b27c4529529be8e17d0bfae1e6b6ecadcab1da6b3341d5b4c` | 0.4951 | 0.4979 | 48.9% (22/45) | -0.1084 |
| **v4_seniority** | Calibrated seniority & core fit | `f868bdbedbe900e7d5243a068079f7c5c0d4efd88110e24a0b82df3740efeed6` | **0.5517** | **0.5681** | **48.9% (22/45)** | **-0.0944** |

*Best Version: `v4_seniority_requirements` (QWK = 0.5517, Exact Agreement = 48.9%). Frozen as canonical judge prompt.*

---

## 2. Confusion Matrices (N = 45)
*Rows = Hand Label (0, 1, 2, 3), Columns = Judge Rounded Prediction (0, 1, 2, 3)*

### Baseline (v1)
| Hand \ Judge | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Total |
|---|---|---|---|---|---|
| **True 0** | 2 | 8 | 3 | 3 | 16 |
| **True 1** | 0 | 0 | 5 | 2 | 7 |
| **True 2** | 1 | 0 | 4 | 5 | 10 |
| **True 3** | 0 | 0 | 3 | 9 | 12 |
| **Total** | 3 | 8 | 15 | 19 | 45 |

### Frozen Best Version (v4)
| Hand \ Judge | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Total |
|---|---|---|---|---|---|
| **True 0** | 7 | 5 | 3 | 1 | 16 |
| **True 1** | 0 | 3 | 2 | 2 | 7 |
| **True 2** | 1 | 0 | 7 | 2 | 10 |
| **True 3** | 0 | 1 | 6 | 5 | 12 |
| **Total** | 8 | 9 | 18 | 10 | 45 |

---

## 3. CV Length Correlation Comparison
*Correlation with candidate CV token count and character count on the 45 tuning pairs.*

| Evaluator | Spearman (Tokens) | p-value | Spearman (Chars) | p-value |
|---|---|---|---|---|
| **Hand Labels (User)** | **-0.1189** | 0.4365 | -0.1109 | 0.4683 |
| **Judge v1 (Baseline)** | **-0.1179** | 0.4407 | -0.1118 | 0.4646 |
| **Judge v4 (Frozen Best)** | **-0.0944** | 0.5367 | -0.0901 | 0.5557 |

*Finding: Both hand labels and the LLM judge exhibit a slight negative, length-neutral correlation with CV length (neither favors longer CVs on these pairs).*

---

## 4. Bias Checks on Tuning Pairs (N = 45)

### Order-Swap Consistency
| Metric | Baseline (v1) | Frozen Best (v4) |
|---|---|---|
| **Original Mean Score** | 2.0901 | 1.6986 |
| **Swapped Mean Score** | 1.8008 | 1.4113 |
| **Mean Signed Diff (Orig - Swapped)** | +0.2893 | +0.2873 |
| **Fraction of Rounded Grade Flips** | 0.4222 (19/45) | 0.4222 (19/45) |
| **Exact Same Rounded Grade** | 57.8% (26/45) | 57.8% (26/45) |

### Per-Role-Family Agreement (v4)
| Role Family | Sample Count (n) | Hand Mean | Judge Mean | Exact Agreement |
|---|---|---|---|---|
| **devops** | 18 | 1.50 | 1.83 | 44.4% (8/18) |
| **javascript** | 18 | 1.28 | 1.76 | 50.0% (9/18) |
| **qa** | 9 | 1.44 | 1.32 | 55.6% (5/9) |
| **Overall** | 45 | 1.40 | 1.70 | 48.9% (22/45) |

---

## 5. Frozen Prompt Template (Version 4)
- **SHA-256 Hash:** `f868bdbedbe900e7d5243a068079f7c5c0d4efd88110e24a0b82df3740efeed6`
- **File:** `data/processed/frozen_judge_prompt_template.txt`

```text
You are an expert technical recruiter evaluating a candidate's CV against a Job Description.
Score the candidate match from 0 to 3 based on core technical competencies and experience:
0: Irrelevant. Candidate has no meaningful alignment with the primary technical requirements or engineering domain.
1: Marginally Relevant. Candidate has minor skill overlap or junior experience, but lacks the core language, primary framework, or necessary seniority.
2: Somewhat Relevant. Candidate satisfies the core technical requirements and primary programming language, demonstrating capable competency to perform the role.
3: Highly Relevant. Candidate is a strong, complete match for the position, with deep relevant experience in the core stack and key responsibilities.

Rules: Evaluate purely on technical fit. You MUST ignore name, gender, age, graduation year, nationality, institution prestige, location, remote preferences, and compensation.

Job Title: {title}
Job Description:
{desc}

CV:
{cv}

Output ONLY a single integer (0, 1, 2, or 3) representing the score.
```

---

## 6. Order-Averaged Diagnostics (45 Tuning Pairs)
*Aggregation rule: Score = (Score[Job->CV] + Score[CV->Job]) / 2*

### Agreement Metrics Comparison
| Metric | Single-Order (v4) | Order-Averaged (v4) |
|---|---|---|
| **Quadratic Weighted Kappa (QWK)** | 0.5517 | **0.6593** |
| **Spearman Rank Correlation** | 0.5681 | **0.6486** |
| **Exact Agreement** | 48.9% (22/45) | **51.1% (23/45)** |
| **Within-One-Grade Rate** | 82.2% (37/45) | **88.9% (40/45)** |
| **Differing by $\ge 2$ Grades** | 8 pairs | **5 pairs** |

### Per-Job Metrics (Order-Averaged)
| Job ID | Role Family / Title | n | Spearman (Avg) | NDCG@10 (Avg) |
|---|---|---|---|---|
| `105723_job` | devops / DevOps Engineer | 9 | 0.8463 | 0.9734 |
| `110001_job` | javascript / Full Stack Node.js/React | 9 | 0.4383 | 0.9840 |
| `26348_job` | javascript / Senior React Native | 9 | -0.2070 | 0.4911 |
| `53525_job` | javascript / Junior FrontEnd Developer | 9 | 0.5798 | 0.8295 |
| `75798_job` | qa / Senior QA Automation (Python) | 9 | 0.7711 | 0.9563 |
| **Mean** | - | 45 | **0.4857** | **0.8469** |

### Bootstrap 95% Confidence Intervals (1,000 resamples)
*(Optimistic estimate: best of 4 prompt versions on 45 pairs)*
- **Spearman:** [0.3171, 0.7546]
- **QWK:** [0.3208, 0.7317]

### Pairs Differing by 2+ Grades
- **Single-Order (8 pairs):**
  1. `53525_job_169636_cv`: Hand = 1, Judge = 3
  2. `53525_job_51549_cv`: Hand = 0, Judge = 2
  3. `53525_job_184226_cv`: Hand = 1, Judge = 3
  4. `26348_job_15790_cv`: Hand = 0, Judge = 2
  5. `26348_job_110565_cv`: Hand = 0, Judge = 3
  6. `26348_job_20564_cv`: Hand = 2, Judge = 0
  7. `26348_job_190676_cv`: Hand = 0, Judge = 2
  8. `75798_job_12268_cv`: Hand = 3, Judge = 1
- **Order-Averaged (5 pairs):**
  1. `53525_job_51549_cv`: Hand = 0, Judge = 2
  2. `26348_job_15790_cv`: Hand = 0, Judge = 2
  3. `26348_job_110565_cv`: Hand = 0, Judge = 2
  4. `26348_job_20564_cv`: Hand = 2, Judge = 0
  5. `75798_job_12268_cv`: Hand = 3, Judge = 1

---

## 7. Held-Out Evaluation Details (41 Pairs, Frozen v4 Order-Averaged)
- **Labels File:** `data/processed/labels_heldout.json`
- **SHA-256:** `5f852d903f82464b8d1b95d5588ac7ca7050693623c23c798097db47f077201d` (verified matches expected).

### Overall Metrics
| Metric | Single-Order (v4) | Order-Averaged (v4) |
|---|---|---|
| **Quadratic Weighted Kappa (QWK)** | 0.3972 | **0.3734** |
| **Spearman Rank Correlation** | 0.5429 | **0.5991** |
| **Exact Agreement** | 34.1% (14/41) | **31.7% (13/41)** |
| **Within-One-Grade Rate** | 82.9% (34/41) | **85.4% (35/41)** |
| **Mean Per-Job Spearman** | 0.3012 | **0.3513** |
| **Mean Per-Job NDCG@10** | 0.9020 | **0.9200** |
| **Random-Order Expected NDCG@10** | 0.8396 | **0.8396** |

### Confusion Matrix (Rows = Hand Label, Cols = Judge Prediction)
| Hand \ Judge | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Total |
|---|---|---|---|---|---|
| **True 0** | 3 | 2 | 4 | 0 | 9 |
| **True 1** | 1 | 0 | 6 | 1 | 8 |
| **True 2** | 0 | 5 | 6 | 0 | 11 |
| **True 3** | 1 | 0 | 8 | 4 | 13 |
| **Total** | 5 | 7 | 24 | 5 | 41 |

### Per-Job Metrics (Held-Out)
| Job ID | Role Family / Title | n | Spearman (Avg) | NDCG@10 (Avg) | Expected Random NDCG@10 |
|---|---|---|---|---|---|
| `129445_job` | other / Shopify Developer | 8 | 0.0514 | 0.8523 | 0.8420 |
| `22047_job` | data / Data Scientist | 8 | 0.6598 | 0.9859 | 0.8608 |
| `50223_job` | java / Java Developer (Krakow) | 9 | 0.8231 | 0.9840 | 0.7684 |
| `64381_job` | qa / Manual QA Engineer | 8 | -0.3254 | 0.7945 | 0.8239 |
| `70278_job` | .net / Middle Full Stack .Net | 8 | 0.5477 | 0.9831 | 0.9031 |
| **Mean** | - | 41 | **0.3513** | **0.9200** | **0.8396** |

### Per-Role-Family Agreement (Held-Out)
| Role Family | n | Hand Mean | Judge Mean | Exact Agreement | Within-One-Grade Rate |
|---|---|---|---|---|---|
| **.net** | 8 | 2.25 | 1.88 | 50.0% (4/8) | 100.0% (8/8) |
| **data** | 8 | 1.62 | 2.12 | 50.0% (4/8) | 75.0% (6/8) |
| **java** | 9 | 1.44 | 1.33 | 22.2% (2/9) | 100.0% (9/9) |
| **other** | 8 | 1.25 | 1.62 | 37.5% (3/8) | 62.5% (5/8) |
| **qa** | 8 | 1.50 | 1.62 | 0.0% (0/8) | 87.5% (7/8) |
| **Overall** | 41 | 1.61 | 1.71 | 31.7% (13/41) | 85.4% (35/41) |

---

## 8. Exact Titles for 8 Flagged Job IDs
| Job ID | Exact Position / Title | Role Family | Primary Keyword | Company Name |
|---|---|---|---|---|
| `2070` | Algorithmic Trading Manager | data | Data Science | Jax.Network |
| `28117` | Electronics Engineer (PCB Designer) | c++ | C++ | agrolabs.io |
| `30297` | FinOps Analyst | data | Data Analyst | Metawork |
| `48467` | IT Security Presales Engineer in VAD Distributor | security | Security | SOFTPROM |
| `88628` | Phyton, .NET, UX designers, Salesmanagers++ | python | Python | SDS Manager Inc |
| `140972` | Website Content Management Specialist | javascript | JavaScript | Lugera |
| `61842` | Lead QA for US product (Design tool) | qa | QA | Visme |
| `95188` | QA Analyst | qa automation | QA Automation | UTOR |
