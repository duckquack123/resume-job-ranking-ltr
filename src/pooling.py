import argparse
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch

def get_bm25_topk(query, corpus_tokens, bm25_model, k):
    query_tokens = query.lower().split()
    scores = bm25_model.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return top_indices

def get_dense_topk(query, corpus_embeddings, model, k):
    query_emb = model.encode([query], convert_to_tensor=True)
    # Cosine similarity
    cos_scores = torch.nn.functional.cosine_similarity(query_emb, corpus_embeddings)
    top_indices = torch.topk(cos_scores, k=k).indices.cpu().numpy()
    return top_indices

def run_pooling(k=5):
    print(f"=== RUNNING POOLING DRY RUN (K={k}) ===")
    out_dir = "/csehome/m25csa007/m25csa007/rjm-ltr/data/processed"
    
    # Use dev set for dry run (take first 10 for timing run)
    jobs_dev = pd.read_parquet(f"{out_dir}/jobs_dev.parquet").head(10)
    cvs_eval = pd.read_parquet(f"{out_dir}/cvs_eval.parquet")
    
    import os
    excluded_cvs = set()
    exclusion_file = f"{out_dir}/excluded_cv_ids.txt"
    if os.path.exists(exclusion_file):
        with open(exclusion_file, 'r') as f:
            excluded_cvs = set(line.strip() for line in f if line.strip())
    
    cvs_eval = cvs_eval[~cvs_eval['id'].isin(excluded_cvs)]
    
    # We might have thousands of CVs in eval.
    # To keep the dry run fast, we'll sample if needed, but since it's just 10 jobs,
    # embedding 20k CVs takes a couple of minutes on A30. Let's do it on the first 10,000 for dry run.
    cvs_eval = cvs_eval.head(10000).reset_index(drop=True)
    
    cv_texts = cvs_eval['CV'].fillna("").tolist()
    cv_ids = cvs_eval['id'].tolist()
    
    print("Preparing BM25...")
    tokenized_corpus = [doc.lower().split() for doc in cv_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("Preparing Dense 1 (bge-small-en-v1.5)...")
    bge_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    bge_corpus = bge_model.encode(cv_texts, convert_to_tensor=True, show_progress_bar=True)
    
    print("Preparing Dense 2 (e5-small-v2)...")
    # For E5, prefix queries with 'query: ' and passages with 'passage: '
    e5_model = SentenceTransformer('intfloat/e5-small-v2')
    e5_passages = ["passage: " + t for t in cv_texts]
    e5_corpus = e5_model.encode(e5_passages, convert_to_tensor=True, show_progress_bar=True)
    
    union_sizes = []
    
    for idx, job_row in jobs_dev.iterrows():
        query = str(job_row['Long Description'])
        
        # BM25
        bm25_indices = get_bm25_topk(query, tokenized_corpus, bm25, k)
        
        # BGE
        bge_indices = get_dense_topk(query, bge_corpus, bge_model, k)
        
        # E5
        e5_query = "query: " + query
        e5_indices = get_dense_topk(e5_query, e5_corpus, e5_model, k)
        
        union_indices = set(bm25_indices).union(set(bge_indices)).union(set(e5_indices))
        union_sizes.append(len(union_indices))
        
        print(f"Job {job_row['id'][:8]} | Union Size: {len(union_indices)} (Max possible: {3*k})")
        
    print(f"\nAverage Union Size across {len(jobs_dev)} dry-run jobs (K={k}): {np.mean(union_sizes):.1f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Pooling depth K per system")
    args = parser.parse_args()
    
    run_pooling(k=args.k)
