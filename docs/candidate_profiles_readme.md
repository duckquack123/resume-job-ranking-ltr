---
language:
- en
license: mit
size_categories:
- 100K<n<1M
dataset_info:
  features:
  - name: Position
    dtype: string
  - name: Moreinfo
    dtype: string
  - name: Looking For
    dtype: string
  - name: Highlights
    dtype: string
  - name: Primary Keyword
    dtype: string
  - name: English Level
    dtype: string
  - name: Experience Years
    dtype: float64
  - name: CV
    dtype: string
  - name: CV_lang
    dtype: string
  - name: id
    dtype: string
  - name: __index_level_0__
    dtype: int64
  splits:
  - name: train
    num_bytes: 425423208
    num_examples: 210250
  download_size: 237415736
  dataset_size: 425423208
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---


# Djinni Dataset (English CVs part)

## Overview
 The [Djinni Recruitment Dataset](https://github.com/Stereotypes-in-LLMs/recruitment-dataset) (English CVs part)  contains 150,000 job descriptions and 230,000 anonymized candidate CVs, posted between 2020-2023 on the [Djinni](https://djinni.co/) IT job platform. The dataset includes samples in English and Ukrainian.

 The dataset contains various attributes related to candidate CVs, including position titles, candidate information, candidate highlights, job search preferences, job profile types, English proficiency levels, experience years, concatenated CV text, language of CVs, and unique identifiers.

## Intended Use

The Djinni dataset is designed with versatility in mind, supporting a wide range of applications:

- **Recommender Systems and Semantic Search:** It serves as a key resource for enhancing job recommendation engines and semantic search functionalities, making the job search process more intuitive and tailored to individual preferences.

- **Advancement of Large Language Models (LLMs):** The dataset provides invaluable training data for both English and Ukrainian domain-specific LLMs. It is instrumental in improving the models' understanding and generation capabilities, particularly in specialized recruitment contexts.

- **Fairness in AI-assisted Hiring:** By serving as a benchmark for AI fairness, the Djinni dataset helps mitigate biases in AI-assisted recruitment processes, promoting more equitable hiring practices.

- **Recruitment Automation:** The dataset enables the development of tools for automated creation of resumes and job descriptions, streamlining the recruitment process.

- **Market Analysis:** It offers insights into the dynamics of Ukraine's tech sector, including the impacts of conflicts, aiding in comprehensive market analysis.

- **Trend Analysis and Topic Discovery:** The dataset facilitates modeling and classification for trend analysis and topic discovery within the tech industry.

- **Strategic Planning:** By enabling the automatic identification of company domains, the dataset assists in strategic market planning.

## Load Dataset

```python
from datasets import load_dataset

data = load_dataset("lang-uk/recruitment-dataset-candidate-profiles-english")['train']
```

## BibTeX entry and citation info
*When publishing results based on this dataset please refer to:*
```bibtex
@inproceedings{drushchak-romanyshyn-2024-introducing,
    title = "Introducing the Djinni Recruitment Dataset: A Corpus of Anonymized {CV}s and Job Postings",
    author = "Drushchak, Nazarii  and
      Romanyshyn, Mariana",
    editor = "Romanyshyn, Mariana  and
      Romanyshyn, Nataliia  and
      Hlybovets, Andrii  and
      Ignatenko, Oleksii",
    booktitle = "Proceedings of the Third Ukrainian Natural Language Processing Workshop (UNLP) @ LREC-COLING 2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.unlp-1.2",
    pages = "8--13",
}
```

## Attribution

Special thanks to [Djinni](https://djinni.co/) for providing this invaluable dataset. Their contribution is crucial in advancing research and development in AI, machine learning, and the broader tech industry. Their effort in compiling and sharing this dataset is greatly appreciated by the community.