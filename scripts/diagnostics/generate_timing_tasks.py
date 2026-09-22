import pandas as pd
import random
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def get_jaccard(text1, text2):
    import re
    def get_shingles(t):
        if not isinstance(t, str): return set()
        words = [w for w in re.split(r'\W+', str(t).lower()) if w]
        return set([' '.join(words[i:i+5]) for i in range(max(1, len(words)-4))])
    
    s1 = get_shingles(text1)
    s2 = get_shingles(text2)
    if not s1 or not s2: return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def main():
    print("Loading TRAIN data...")
    jobs = pd.read_parquet("data/processed/jobs_train.parquet")
    cvs = pd.read_parquet("data/processed/cvs_train.parquet")
    
    # Take first 15 jobs to get roughly 100 pairs
    jobs = jobs.head(15)
    # Only need a subset of CVs to speed up the dummy retrieval for this test
    # We will use 10,000 CVs instead of 168k to save memory/time just for task generation
    cvs = cvs.head(10000)
    
    cv_texts = cvs['CV'].fillna("").tolist()
    cv_ids = cvs['id'].tolist()
    
    print("Encoding CVs (BM25)...")
    tokenized_corpus = [str(doc).lower().split() for doc in cv_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("Encoding CVs (BGE)...")
    bge_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    bge_corpus = bge_model.encode(cv_texts, convert_to_tensor=True, show_progress_bar=False)
    
    print("Encoding CVs (E5)...")
    e5_model = SentenceTransformer('intfloat/e5-small-v2')
    e5_passages = ["passage: " + str(t) for t in cv_texts]
    e5_corpus = e5_model.encode(e5_passages, convert_to_tensor=True, show_progress_bar=False)
    
    K = 3
    tasks = []
    
    for _, job_row in jobs.iterrows():
        jid = job_row['id']
        jtitle = str(job_row['Position'])
        jdesc = str(job_row['Long Description'])
        
        # BM25
        query_tokens = str(jdesc).lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_idx = np.argsort(bm25_scores)[::-1][:K]
        
        # BGE
        bge_emb = bge_model.encode([jdesc], convert_to_tensor=True, show_progress_bar=False)
        bge_scores = torch.nn.functional.cosine_similarity(bge_emb, bge_corpus)
        bge_idx = torch.topk(bge_scores, k=K).indices.cpu().numpy()
        
        # E5
        e5_q = "query: " + jdesc
        e5_emb = e5_model.encode([e5_q], convert_to_tensor=True, show_progress_bar=False)
        e5_scores = torch.nn.functional.cosine_similarity(e5_emb, e5_corpus)
        e5_idx = torch.topk(e5_scores, k=K).indices.cpu().numpy()
        
        pool_cvs = {}
        for i in list(bm25_idx) + list(bge_idx) + list(e5_idx):
            cid = cv_ids[i]
            if cid not in pool_cvs:
                pool_cvs[cid] = cv_texts[i]
                
        final_pool = []
        for cid, txt in pool_cvs.items():
            is_dupe = False
            for prev_cid in final_pool:
                if get_jaccard(txt, pool_cvs[prev_cid]) >= 0.70:
                    is_dupe = True
                    break
            if not is_dupe:
                final_pool.append(cid)
                
        for cid in final_pool:
            if len(tasks) < 100:
                tasks.append({
                    "pair_id": f"{jid}_{cid}",
                    "job_title": jtitle,
                    "job_desc": jdesc,
                    "cv_text": pool_cvs[cid]
                })
        
        if len(tasks) >= 100:
            break
            
    # Ensure exactly 100
    tasks = tasks[:100]
    
    with open("data/processed/timing_tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)
        
    print(f"Generated {len(tasks)} tasks for timing test.")

if __name__ == "__main__":
    main()
