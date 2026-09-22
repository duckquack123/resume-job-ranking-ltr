import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import scipy.stats as st

def preprocess_bm25(text):
    if not isinstance(text, str): return []
    text = text.lower()
    tokens = re.findall(r'\b[a-z]{2,}\b', text)
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return tokens

def get_bm25_topk(query_tokens, bm25_model, k):
    scores = bm25_model.get_scores(query_tokens)
    return np.argsort(scores)[::-1][:k]

def get_dense_topk(query, corpus_embeddings, model, k):
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    cos_scores = torch.nn.functional.cosine_similarity(query_emb, corpus_embeddings)
    return torch.topk(cos_scores, k=k).indices.cpu().numpy()

def overlap(l1, l2):
    return len(set(l1).intersection(set(l2)))

def cluster_bootstrap_ci(data, cluster_ids, n_bootstraps=1000, confidence=0.95):
    import pandas as pd
    df = pd.DataFrame({'val': data, 'cluster': cluster_ids})
    cluster_sums = df.groupby('cluster')['val'].sum().values
    cluster_counts = df.groupby('cluster')['val'].count().values
    n_clusters = len(cluster_sums)
    bootstrapped_means = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        idx = np.random.choice(n_clusters, size=n_clusters, replace=True)
        bootstrapped_means[i] = cluster_sums[idx].sum() / cluster_counts[idx].sum()
    lower = np.percentile(bootstrapped_means, 2.5)
    upper = np.percentile(bootstrapped_means, 97.5)
    return df['val'].mean(), lower, upper

def run_scale_up():
    out_dir = "/csehome/m25csa007/m25csa007/rjm-ltr/data/processed"
    import hashlib
    print("Loading data...")
    
    # Hash check
    with open(f"{out_dir}/jobs_train.parquet", "rb") as f: jobs_bytes = f.read()
    with open(f"{out_dir}/cvs_train.parquet", "rb") as f: cvs_bytes = f.read()
    print(f"jobs_train.parquet SHA-256: {hashlib.sha256(jobs_bytes).hexdigest()}")
    print(f"cvs_train.parquet SHA-256: {hashlib.sha256(cvs_bytes).hexdigest()}")
    
    jobs_train = pd.read_parquet(f"{out_dir}/jobs_train.parquet").sample(n=100, random_state=42).reset_index(drop=True)
    cvs_train = pd.read_parquet(f"{out_dir}/cvs_train.parquet")
    
    import os
    excluded_cvs = set()
    exclusion_file = f"{out_dir}/excluded_cv_ids.txt"
    if os.path.exists(exclusion_file):
        with open(exclusion_file, 'r') as f:
            excluded_cvs = set(line.strip() for line in f if line.strip())
    
    cvs_train = cvs_train[~cvs_train['id'].isin(excluded_cvs)]
    print(f"Sampled exactly {len(jobs_train)} jobs.")
    
    cv_texts = cvs_train['CV'].fillna("").tolist()
    cv_keywords = cvs_train['Primary Keyword'].fillna("").str.lower().tolist()
    
    # Check for resume
    results_file = f"{out_dir}/pooling_results.jsonl"
    processed_ids = set()
    import json
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data['job_id'])
        print(f"Resuming: found {len(processed_ids)} already processed jobs.")
    
    if len(processed_ids) == len(jobs_train):
        print("All jobs processed. Displaying metrics...")
    else:
        # 2. Truncation Stats (using 10k random samples)
        from transformers import AutoTokenizer
        import random
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
        
        print("\n=== TRUNCATION STATS (10k sample) ===")
        sample_cvs = random.sample(cv_texts, min(10000, len(cv_texts)))
        all_jobs_train = pd.read_parquet(f"{out_dir}/jobs_train.parquet")
        sample_jobs = all_jobs_train['Long Description'].fillna("").sample(n=min(10000, len(all_jobs_train)), random_state=42).tolist()
        
        cv_lens = [len(tokenizer.encode(t, add_special_tokens=True)) for t in sample_cvs]
        job_lens = [len(tokenizer.encode(t, add_special_tokens=True)) for t in sample_jobs]
        
        print(f"CVs > 512 tokens: {sum(1 for l in cv_lens if l > 512)/len(cv_lens):.1%}")
        print(f"CVs Percentiles (50/90/99): {np.percentile(cv_lens, [50, 90, 99])}")
        print(f"Jobs > 512 tokens: {sum(1 for l in job_lens if l > 512)/len(job_lens):.1%}")
        print(f"Jobs Percentiles (50/90/99): {np.percentile(job_lens, [50, 90, 99])}")
        
        # Setup Encoders
        print("\nPreparing Encoders for scale-up (N=" + str(len(cv_texts)) + ")...")
        tokenized_corpus = [preprocess_bm25(doc) for doc in cv_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        bge_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        bge_corpus = bge_model.encode(cv_texts, convert_to_tensor=True, show_progress_bar=True)
        
        e5_model = SentenceTransformer('intfloat/e5-small-v2')
        e5_passages = ["passage: " + t for t in cv_texts]
        e5_corpus = e5_model.encode(e5_passages, convert_to_tensor=True, show_progress_bar=True)
        
        k = 5
        
        print("\nRunning queries...")
        with open(results_file, 'a') as f:
            for idx, job_row in jobs_train.iterrows():
                job_id = str(job_row['id'])
                if job_id in processed_ids:
                    continue
                    
                query = str(job_row['Long Description'])
                job_title = str(job_row['Position'])
                job_keyword = str(job_row['Primary Keyword']).lower().strip()
                
                # 3. Tested explanation: how many CVs in pool share keyword?
                pool_match_count = cv_keywords.count(job_keyword)
                density = pool_match_count / len(cv_keywords)
                
                bm25_q = preprocess_bm25(job_title + " " + query)
                bm25_indices = get_bm25_topk(bm25_q, bm25, k)
                
                bge_q = "Represent this sentence for searching relevant passages: " + query
                bge_indices = get_dense_topk(bge_q, bge_corpus, bge_model, k)
                
                e5_q = "query: " + query
                e5_indices = get_dense_topk(e5_q, e5_corpus, e5_model, k)
                
                bm25_bge_overlap = overlap(bm25_indices, bge_indices) / k
                bm25_e5_overlap = overlap(bm25_indices, e5_indices) / k
                bge_e5_overlap = overlap(bge_indices, e5_indices) / k
                
                union_indices = set(bm25_indices).union(set(bge_indices)).union(set(e5_indices))
                union_size = len(union_indices)
                
                match_bm25 = sum(1 for c in bm25_indices if cv_keywords[c] == job_keyword) / k
                match_bge = sum(1 for c in bge_indices if cv_keywords[c] == job_keyword) / k
                match_e5 = sum(1 for c in e5_indices if cv_keywords[c] == job_keyword) / k
                
                res = {
                    "job_id": job_id,
                    "density": density,
                    "bm25_bge_overlap": bm25_bge_overlap,
                    "bm25_e5_overlap": bm25_e5_overlap,
                    "bge_e5_overlap": bge_e5_overlap,
                    "union_size": union_size,
                    "match_bm25": match_bm25,
                    "match_bge": match_bge,
                    "match_e5": match_e5
                }
                f.write(json.dumps(res) + '\n')
                f.flush()

    # Read back all results
    bm25_bge_overlap = []
    bm25_e5_overlap = []
    bge_e5_overlap = []
    union_sizes = []
    match_bm25, match_bge, match_e5 = [], [], []
    pool_keyword_density = []
    job_ids = []
    
    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            job_ids.append(data['job_id'])
            bm25_bge_overlap.append(data['bm25_bge_overlap'])
            bm25_e5_overlap.append(data['bm25_e5_overlap'])
            bge_e5_overlap.append(data['bge_e5_overlap'])
            union_sizes.append(data['union_size'])
            match_bm25.append(data['match_bm25'])
            match_bge.append(data['match_bge'])
            match_e5.append(data['match_e5'])
            pool_keyword_density.append(data['density'])
            
    print("\n=== SCALE UP DIAGNOSTICS (100 Train Jobs) ===")
    jobs_train = pd.read_parquet(f"{out_dir}/jobs_train.parquet")
    jobs_train['job_id'] = jobs_train['id'].astype(str)
    res_df = pd.DataFrame({'job_id': job_ids})
    res_df = res_df.merge(jobs_train[['job_id', 'Company Name']], on='job_id', how='left')
    company_clusters = res_df['Company Name'].tolist()
    
    def print_ci(name, data, is_pct=True):
        m, l, u = cluster_bootstrap_ci(data, company_clusters)
        if is_pct:
            print(f"{name}: {m:.1%} [95% CI: {l:.1%} - {u:.1%}]")
        else:
            print(f"{name}: {m:.1f} [95% CI: {l:.1f} - {u:.1f}]")
            
    print_ci("Union Size", union_sizes, is_pct=False)
    print_ci("BM25-BGE Overlap@5", bm25_bge_overlap)
    print_ci("BM25-E5  Overlap@5", bm25_e5_overlap)
    print_ci("BGE-E5   Overlap@5", bge_e5_overlap)
    
    print("\nPrimary Keyword Match Rate:")
    print_ci("BM25", match_bm25)
    print_ci("BGE ", match_bge)
    print_ci("E5  ", match_e5)
    
    print("\nTested Explanation for Low Overlap (Keyword Density):")
    density_mean, density_l, density_u = cluster_bootstrap_ci(pool_keyword_density, company_clusters)
    print(f"Mean % of CVs in pool matching Job's Keyword: {density_mean:.2%} [95% CI: {density_l:.2%} - {density_u:.2%}]")
    
if __name__ == "__main__":
    import random
    import os
    run_scale_up()
