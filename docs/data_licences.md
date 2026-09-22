# Data Licences and Dataset Audit

> **Status:** Phase 0 audit. Updated 2026-09-20 following Fallback approval.

---

## 1. Candidate Dataset Audit (Decisions Applied)

### 1.1 TalentCLEF 2026 Task A (Zero-Shot Eval Only)

| Field | Details |
|---|---|
| **Source** | Zenodo DOI: [10.5281/zenodo.19652670](https://zenodo.org/records/19652670). Confirmed this is the official release via the CLEF 2026 website and official tutorials. |
| **Status** | **REJECTED as training set. KEPT as zero-shot external evaluation.** Only 50 queries total across dev and test. |
| **Licence** | **Creative Commons Attribution 4.0 International (CC BY 4.0)** (Standard open access on Zenodo). |
| **Size** | 10 queries/472 resumes (dev) + 40 queries/476 resumes (test) = 50 queries, 948 resumes total. |

---

### 1.2 Main Benchmark Sources (Fallback Plan)

We will construct the main training and evaluation dataset from publicly available, open-licence sources:

#### A. Resumes
| Field | Details |
|---|---|
| **Dataset** | `lang-uk/recruitment-dataset-candidate-profiles-english` (Hugging Face) |
| **Licence** | **MIT Licence** |
| **Role** | Main candidate pool. Real resumes. Synthetic resumes will not be used in the main pool (only as a tagged augmentation condition). |
| **English Row Count** | 210,250 CVs |
| **Schema** | `Position`, `Moreinfo`, `Looking For`, `Highlights`, `Primary Keyword`, `English Level`, `Experience Years`, `CV`, `CV_lang`, `id`, `__index_level_0__` |

#### B. Job Descriptions
| Field | Details |
|---|---|
| **Dataset** | `lang-uk/recruitment-dataset-job-descriptions-english` (Hugging Face) |
| **Licence** | **MIT Licence** |
| **Role** | Query set. |
| **English Row Count** | 141,897 Job Descriptions |
| **Schema** | `Position`, `Long Description`, `Company Name`, `Exp Years`, `Primary Keyword`, `English Level`, `Published`, `Long Description_lang`, `id`, `__index_level_0__` |

---

## 2. LLM Judge and Model Families

To enforce strict independence (R12), we separate the model families:
- **LLM Re-Ranker (Testing):** `Qwen2.5-7B-Instruct` (Apache 2.0)
- **LLM Judge (Training/Val Labels):** Must be a different family. Proposed: `Llama-3.1-8B-Instruct` (Llama 3.1 Community License). 
- *Licence Check constraint:* We will verify that training downstream models (LambdaMART / bi-encoders) on the outputs of the chosen LLM Judge does not violate its specific acceptable use policy regarding training other models.

*No labels will be derived from retrieval scores. Weak labeling is rejected.*

---

## 3. Human Annotation Gold Set

- **Size:** ~100 jobs × 15–20 pooled candidates.
- **Privacy:** All data and annotation stays local. No cloud tools.
- **Goal:** Validating the LLM Judge before using it to generate training labels.
