import os
import re
import json
import time
import torch
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Limit CPU thread contention
torch.set_num_threads(4)

# Set HF token
if not os.environ.get("HF_TOKEN"):
    hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(hf_token_path):
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()
    else:
        env_file = "/csehome/m25csa007/m25csa007/my.env"
        if os.path.exists(env_file):
            for line in open(env_file):
                if "HF_TOKEN" in line:
                    os.environ["HF_TOKEN"] = line.split("=")[-1].strip().strip("\"'")

TECH_TERMS = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang", "rust", "php", 
    "ruby", "swift", "kotlin", "scala", "r", "perl", "bash", "shell", "powershell", "dart", 
    "elixir", "clojure", "haskell", "julia", "lua", "html", "css", "sql",
    # Frontend Frameworks & Libraries
    "react", "angular", "vue", "vue.js", "svelte", "next.js", "nuxt", "redux", "mobx", "jquery", 
    "tailwind", "bootstrap", "material-ui", "mui", "webpack", "vite", "babel",
    # Backend Frameworks & Libraries
    "django", "flask", "fastapi", "spring", "spring boot", "express", "express.js", "nestjs", 
    "laravel", "symfony", "rails", "ruby on rails", "asp.net", ".net", "fastify", "koa", "gin",
    # Databases & Caching
    "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis", "elasticsearch", "opensearch", 
    "cassandra", "dynamodb", "couchdb", "neo4j", "mariadb", "oracle", "mssql", "clickhouse", 
    "snowflake", "bigquery", "kafka", "rabbitmq",
    # Cloud & DevOps / Infrastructure
    "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins", 
    "gitlab", "github actions", "circleci", "helm", "prometheus", "grafana", "nginx", "apache", 
    "linux", "ci/cd", "git",
    # AI / ML / Data
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy", "spark", "pyspark", 
    "hadoop", "airflow", "dbt", "huggingface", "llm", "langchain", "opencv", "nlp",
    # Mobile / Systems / API
    "android", "ios", "flutter", "react native", "unity", "unreal", "graphql", "grpc", "rest", "restful"
]

# Precompile regexes for fast matching
COMPILED_TECH_REGEX = [
    (term, re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE))
    for term in TECH_TERMS
]

def extract_tech_terms(text):
    if not text:
        return set()
    text = str(text)
    found = set()
    for term, regex in COMPILED_TECH_REGEX:
        if regex.search(text):
            found.add(term)
    return found

def compute_within_job_features(df, group_col='job_id'):
    """
    Computes within-job rank (1 = top score) and score minus top score for continuous features.
    """
    cols_to_norm = ['bm25', 'bge', 'e5', 'tech_overlap', 'cv_token_len']
    
    df_out = df.copy()
    for col in cols_to_norm:
        if col not in df_out.columns:
            continue
        rank_col = f"{col}_rank"
        diff_col = f"{col}_diff_top"
        
        # Rank: rankdata(-val, method='min') so highest score gets rank 1
        df_out[rank_col] = df_out.groupby(group_col)[col].transform(
            lambda s: rankdata(-s.values, method='min')
        )
        # Score minus top score: score - max_score (<= 0.0)
        df_out[diff_col] = df_out.groupby(group_col)[col].transform(
            lambda s: s - s.max()
        )
    return df_out

def process_dev_features():
    print("\n=== Processing 86 Dev Features ===", flush=True)
    t0 = time.time()
    with open("data/processed/dev_hand_labeling_tasks.json") as f:
        dev_tasks = json.load(f)
    
    with open("data/processed/labels_tuning.json") as f:
        tuning_labels = {p["pair_id"].strip(): p["score"] for p in json.load(f)}
        
    with open("data/processed/labels_heldout.json") as f:
        heldout_labels = {p["pair_id"].strip(): p["score"] for p in json.load(f)}
        
    with open("data/processed/dev_hand_labeling_provenance.json") as f:
        dev_prov = json.load(f)

    tasks_cleaned = []
    for t in dev_tasks:
        raw_pid = t["pair_id"].strip().replace("\n", "")
        parts = raw_pid.split("_job_")
        jid = parts[0] + "_job" if not parts[0].endswith("_job") else parts[0]
        cid = parts[1] + "_cv" if not parts[1].endswith("_cv") else parts[1]
        
        lbl = tuning_labels.get(raw_pid, heldout_labels.get(raw_pid, None))
        split = "tuning" if raw_pid in tuning_labels else ("heldout" if raw_pid in heldout_labels else "unknown")
        
        tasks_cleaned.append({
            "pair_id": raw_pid,
            "job_id": jid,
            "cv_id": cid,
            "job_title": t["job_title"],
            "job_desc": t["job_desc"],
            "cv_text": t["cv_text"],
            "split": split,
            "human_label": lbl,
            "provenance": dev_prov.get(raw_pid, [])
        })

    # 1. BM25 on cvs_dev corpus
    print("Building BM25 on cvs_dev corpus (21k CVs)...", flush=True)
    cvs_dev = pd.read_parquet("data/processed/cvs_dev.parquet")
    with open("data/processed/excluded_cv_ids.txt") as f:
        excluded = set(x.strip() for x in f if x.strip())
    cvs_dev = cvs_dev[~cvs_dev["id"].isin(excluded)].reset_index(drop=True)
    cv_texts = cvs_dev["CV"].fillna("").tolist()
    cv_ids = cvs_dev["id"].tolist()
    cv_id_to_idx = {cid: idx for idx, cid in enumerate(cv_ids)}
    
    tokenized_corpus = [str(doc).lower().split() for doc in cv_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 built in {time.time() - t0:.2f}s", flush=True)
    
    # 2. Dense Models
    print("Loading BGE and E5 models for dev scoring...", flush=True)
    bge_model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
    e5_model = SentenceTransformer('intfloat/e5-small-v2', device='cpu')
    
    unique_jids = list({t["job_id"]: t["job_desc"] for t in tasks_cleaned}.keys())
    unique_jdescs = [{t["job_id"]: t["job_desc"] for t in tasks_cleaned}[jid] for jid in unique_jids]
    
    unique_cids = list({t["cv_id"]: t["cv_text"] for t in tasks_cleaned}.keys())
    unique_ctexts = [{t["cv_id"]: t["cv_text"] for t in tasks_cleaned}[cid] for cid in unique_cids]
    
    print(f"Batch-encoding {len(unique_jids)} dev jobs and {len(unique_cids)} dev CVs...", flush=True)
    # BGE
    bge_job_arr = bge_model.encode(unique_jdescs, batch_size=16, convert_to_tensor=True, show_progress_bar=False)
    bge_cv_arr = bge_model.encode(unique_ctexts, batch_size=32, convert_to_tensor=True, show_progress_bar=False)
    job_bge_emb = {jid: bge_job_arr[i] for i, jid in enumerate(unique_jids)}
    cv_bge_emb = {cid: bge_cv_arr[i] for i, cid in enumerate(unique_cids)}
    
    # E5
    e5_queries = ["query: " + d for d in unique_jdescs]
    e5_passages = ["passage: " + t for t in unique_ctexts]
    e5_job_arr = e5_model.encode(e5_queries, batch_size=16, convert_to_tensor=True, show_progress_bar=False)
    e5_cv_arr = e5_model.encode(e5_passages, batch_size=32, convert_to_tensor=True, show_progress_bar=False)
    job_e5_emb = {jid: e5_job_arr[i] for i, jid in enumerate(unique_jids)}
    cv_e5_emb = {cid: e5_cv_arr[i] for i, cid in enumerate(unique_cids)}
    
    # Tokenizer for CV length
    print("Loading tokenizer for CV token length...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=os.environ.get("HF_TOKEN"))
    cv_tok_lens = {
        cid: len(tokenizer.encode(txt, add_special_tokens=False))
        for cid, txt in zip(unique_cids, unique_ctexts)
    }
    
    # BM25 scores
    job_bm25_scores = {}
    for jid, desc in zip(unique_jids, unique_jdescs):
        q_tokens = str(desc).lower().split()
        job_bm25_scores[jid] = bm25.get_scores(q_tokens)
        
    records = []
    print("Assembling dev features...", flush=True)
    for t in tasks_cleaned:
        jid = t["job_id"]
        cid = t["cv_id"]
        
        # BM25
        c_idx = cv_id_to_idx[cid]
        bm25_val = float(job_bm25_scores[jid][c_idx])
        
        # BGE
        bge_sim = float(torch.nn.functional.cosine_similarity(
            job_bge_emb[jid].unsqueeze(0), cv_bge_emb[cid].unsqueeze(0)
        ).item())
        
        # E5
        e5_sim = float(torch.nn.functional.cosine_similarity(
            job_e5_emb[jid].unsqueeze(0), cv_e5_emb[cid].unsqueeze(0)
        ).item())
        
        # CV token length
        cv_tok_len = cv_tok_lens[cid]
        
        # Tech overlap
        j_tech = extract_tech_terms(t["job_title"] + " " + t["job_desc"])
        c_tech = extract_tech_terms(t["cv_text"])
        overlap_cnt = len(j_tech & c_tech)
        
        records.append({
            "pair_id": t["pair_id"],
            "job_id": jid,
            "cv_id": cid,
            "split": t["split"],
            "human_label": t["human_label"],
            "bm25": bm25_val,
            "bge": bge_sim,
            "e5": e5_sim,
            "cv_token_len": cv_tok_len,
            "tech_overlap": overlap_cnt,
            "provenance": t["provenance"]
        })
        
    df_dev = pd.DataFrame(records)
    df_dev = compute_within_job_features(df_dev, group_col='job_id')
    out_path = "data/processed/features_dev_86.parquet"
    df_dev.to_parquet(out_path, index=False)
    print(f"Saved {len(df_dev)} dev features to {out_path} in {time.time() - t0:.2f}s.", flush=True)
    return df_dev

def process_train_features():
    print("\n=== Processing Train Features (4,283 pairs) ===", flush=True)
    t0 = time.time()
    pairs_path = "data/processed/train_490_pairs_clean.parquet"
    df_pairs = pd.read_parquet(pairs_path)
    print(f"Loaded clean train pairs: {len(df_pairs)} across {df_pairs['job_id'].nunique()} jobs.", flush=True)
    
    # Load texts
    print("Loading job text...", flush=True)
    jobs_df = pd.read_parquet("data/processed/jobs_train.parquet")
    jobs_sub = jobs_df[jobs_df["id"].isin(df_pairs["job_id"].unique())].set_index("id")
    job_titles = jobs_sub["Position"].fillna("").to_dict()
    job_descs = jobs_sub["Long Description"].fillna("").to_dict()
    
    print("Loading CV text...", flush=True)
    cvs_df = pd.read_parquet("data/processed/cvs_train.parquet")
    cvs_sub = cvs_df[cvs_df["id"].isin(df_pairs["cv_id"].unique())].set_index("id")
    cv_texts = cvs_sub["CV"].fillna("").to_dict()
    
    # Tokenizer
    print("Loading tokenizer for CV token length...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=os.environ.get("HF_TOKEN"))
    
    print("Batch-tokenizing 3,270 unique CVs...", flush=True)
    unique_cids = list(df_pairs["cv_id"].unique())
    unique_cv_texts = [cv_texts.get(cid, "") for cid in unique_cids]
    
    # Use tokenizer fast batch encoding
    encodings = tokenizer(unique_cv_texts, add_special_tokens=False, padding=False, truncation=False)
    cv_tok_lens = {cid: len(ids) for cid, ids in zip(unique_cids, encodings["input_ids"])}
    
    print("Extracting technology terms...", flush=True)
    unique_jids = list(df_pairs["job_id"].unique())
    job_tech_map = {jid: extract_tech_terms(str(job_titles.get(jid, "")) + " " + str(job_descs.get(jid, ""))) for jid in unique_jids}
    cv_tech_map = {cid: extract_tech_terms(cv_texts.get(cid, "")) for cid in unique_cids}
    
    # Build dataframe
    df_train = df_pairs.copy()
    df_train.rename(columns={"bm25_score": "bm25", "bge_score": "bge", "e5_score": "e5"}, inplace=True)
    
    df_train["cv_token_len"] = df_train["cv_id"].map(cv_tok_lens)
    df_train["tech_overlap"] = [
        len(job_tech_map[r.job_id] & cv_tech_map[r.cv_id])
        for r in df_train.itertuples()
    ]
    
    df_train = compute_within_job_features(df_train, group_col='job_id')
    
    out_path = "data/processed/features_train.parquet"
    df_train.to_parquet(out_path, index=False)
    print(f"Saved {len(df_train)} train features to {out_path} in {time.time() - t0:.2f}s.", flush=True)
    return df_train

if __name__ == "__main__":
    process_dev_features()
    process_train_features()
