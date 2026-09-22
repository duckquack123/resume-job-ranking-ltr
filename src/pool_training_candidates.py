import os
import json
import torch
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_bm25(text):
    if not isinstance(text, str): return []
    text = text.lower()
    tokens = re.findall(r'\b[a-z]{2,}\b', text)
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique

def get_bm25_topk(query_tokens, bm25_model, k):
    scores = bm25_model.get_scores(query_tokens)
    return np.argsort(scores)[::-1][:k], scores

def get_dense_topk(query, corpus_embeddings, model, k):
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    cos_scores = torch.nn.functional.cosine_similarity(query_emb, corpus_embeddings)
    scores_vals, indices = torch.topk(cos_scores, k=k)
    return indices.cpu().numpy(), scores_vals.cpu().numpy()

def get_5gram_shingles(text):
    if not isinstance(text, str): return set()
    words = [w for w in re.split(r'\W+', text.lower()) if w]
    if len(words) < 5:
        return set([' '.join(words)]) if words else set()
    return set([' '.join(words[i:i+5]) for i in range(len(words)-4)])

def jaccard_similarity(s1, s2):
    if not s1 or not s2: return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def main():
    print("Loading data...")
    jobs_df = pd.read_parquet("data/processed/jobs_train.parquet")
    cvs_df = pd.read_parquet("data/processed/cvs_train.parquet")
    
    with open("data/processed/train_500_job_ids.txt", "r") as f:
        job_ids = [line.strip() for line in f if line.strip()]
        
    # Exclude 114596_cv
    excluded_cvs = set()
    if os.path.exists("data/processed/excluded_cv_ids.txt"):
        with open("data/processed/excluded_cv_ids.txt", "r") as f:
            excluded_cvs = set(line.strip() for line in f if line.strip())
            
    cvs_df = cvs_df[~cvs_df['id'].isin(excluded_cvs)].reset_index(drop=True)
    
    cv_texts = cvs_df['CV'].fillna("").tolist()
    cv_ids = cvs_df['id'].astype(str).tolist()
    
    # Filter to the 500 jobs
    jobs_subset = jobs_df[jobs_df['id'].isin(job_ids)].reset_index(drop=True)
    
    print("Preparing Encoders for pooling (N=" + str(len(cv_texts)) + ")...")
    tokenized_corpus = [preprocess_bm25(doc) for doc in cv_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    cache_dir = "data/processed/cache"
    os.makedirs(cache_dir, exist_ok=True)
    bge_cache = os.path.join(cache_dir, "bge_train_corpus.pt")
    e5_cache = os.path.join(cache_dir, "e5_train_corpus.pt")
    
    bge_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    if os.path.exists(bge_cache):
        print(f"Loading cached BGE corpus embeddings from {bge_cache}...")
        bge_corpus = torch.load(bge_cache, map_location=bge_model.device)
    else:
        print("Computing BGE corpus embeddings...")
        bge_corpus = bge_model.encode(cv_texts, convert_to_tensor=True, show_progress_bar=True)
        torch.save(bge_corpus, bge_cache)
    
    e5_model = SentenceTransformer('intfloat/e5-small-v2')
    if os.path.exists(e5_cache):
        print(f"Loading cached E5 corpus embeddings from {e5_cache}...")
        e5_corpus = torch.load(e5_cache, map_location=e5_model.device)
    else:
        print("Computing E5 corpus embeddings...")
        e5_passages = ["passage: " + t for t in cv_texts]
        e5_corpus = e5_model.encode(e5_passages, convert_to_tensor=True, show_progress_bar=True)
        torch.save(e5_corpus, e5_cache)
    
    k = 3
    pairs_out = []
    total_candidates_removed = 0
    total_candidates_pooled_before_dedup = 0
    
    print("Pooling candidates for 500 jobs (K=3) with near-deduplication (word 5-gram, 0.70)...", flush=True)
    for job_num, (_, row) in enumerate(jobs_subset.iterrows()):
        if (job_num + 1) % 50 == 0 or job_num == 0:
            print(f"Processing job {job_num + 1}/500...", flush=True)
        jid = str(row['id'])
        query = str(row['Long Description'])
        title = str(row['Position'])
        
        # BM25
        bm25_q = preprocess_bm25(title + " " + query)
        bm25_idx, bm25_scores = get_bm25_topk(bm25_q, bm25, k)
        
        # BGE
        bge_q = "Represent this sentence for searching relevant passages: " + query
        bge_idx, bge_scores = get_dense_topk(bge_q, bge_corpus, bge_model, k)
        bge_q_emb = bge_model.encode([bge_q], convert_to_tensor=True, show_progress_bar=False)
        
        # E5
        e5_q = "query: " + query
        e5_idx, e5_scores = get_dense_topk(e5_q, e5_corpus, e5_model, k)
        e5_q_emb = e5_model.encode([e5_q], convert_to_tensor=True, show_progress_bar=False)
        
        # Preserve retrieval order: BM25 top-k, then BGE top-k, then E5 top-k (deduplicate IDs)
        ordered_candidates = []
        for idx_list in [bm25_idx, bge_idx, e5_idx]:
            for idx in idx_list:
                if idx not in ordered_candidates:
                    ordered_candidates.append(idx)
                    
        total_candidates_pooled_before_dedup += len(ordered_candidates)
        
        # Near-deduplicate candidate CVs for this job (word 5-gram, Jaccard >= 0.70)
        accepted_candidates = []
        accepted_shingles = []
        
        for c_idx in ordered_candidates:
            c_text = cv_texts[c_idx]
            c_shingles = get_5gram_shingles(c_text)
            
            is_dupe = False
            for prev_shingles in accepted_shingles:
                if jaccard_similarity(c_shingles, prev_shingles) >= 0.70:
                    is_dupe = True
                    break
                    
            if is_dupe:
                total_candidates_removed += 1
            else:
                accepted_candidates.append(c_idx)
                accepted_shingles.append(c_shingles)
        
        # Compute scores for accepted candidates
        for c_idx in accepted_candidates:
            cid = cv_ids[c_idx]
            c_bm25_score = bm25_scores[c_idx]
            
            c_emb = bge_corpus[c_idx].unsqueeze(0)
            c_bge_score = torch.nn.functional.cosine_similarity(bge_q_emb, c_emb).item()
            
            c_e5_emb = e5_corpus[c_idx].unsqueeze(0)
            c_e5_score = torch.nn.functional.cosine_similarity(e5_q_emb, c_e5_emb).item()
            
            pairs_out.append({
                "job_id": jid,
                "cv_id": cid,
                "bm25_score": float(c_bm25_score),
                "bge_score": float(c_bge_score),
                "e5_score": float(c_e5_score)
            })
            
    df_pairs = pd.DataFrame(pairs_out)
    out_path = "data/processed/train_500_pairs.parquet"
    df_pairs.to_parquet(out_path, index=False)
    
    print(f"\nPooling complete. Saved to {out_path}.")
    print(f"Total jobs: {len(jobs_subset)}")
    print(f"Total candidates pooled before dedup: {total_candidates_pooled_before_dedup}")
    print(f"Candidate CVs removed via near-deduplication (word 5-gram, 0.70): {total_candidates_removed}")
    print(f"Total final pairs: {len(df_pairs)}")

if __name__ == "__main__":
    main()
