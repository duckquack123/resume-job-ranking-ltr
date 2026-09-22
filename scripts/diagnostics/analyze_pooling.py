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
    cos_scores = torch.nn.functional.cosine_similarity(query_emb, corpus_embeddings)
    top_indices = torch.topk(cos_scores, k=k).indices.cpu().numpy()
    return top_indices

def overlap(l1, l2):
    return len(set(l1).intersection(set(l2)))

def run_pooling_analysis(k=5):
    out_dir = "/csehome/m25csa007/m25csa007/rjm-ltr/data/processed"
    jobs_dev = pd.read_parquet(f"{out_dir}/jobs_dev.parquet").head(10)
    cvs_eval = pd.read_parquet(f"{out_dir}/cvs_eval.parquet").head(10000).reset_index(drop=True)
    
    cv_texts = cvs_eval['CV'].fillna("").tolist()
    cv_keywords = cvs_eval['Primary Keyword'].fillna("").str.lower().tolist()
    cv_ids = cvs_eval['id'].tolist()
    
    tokenized_corpus = [doc.lower().split() for doc in cv_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bge_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    bge_corpus = bge_model.encode(cv_texts, convert_to_tensor=True)
    
    e5_model = SentenceTransformer('intfloat/e5-small-v2')
    e5_passages = ["passage: " + t for t in cv_texts]
    e5_corpus = e5_model.encode(e5_passages, convert_to_tensor=True)
    
    bm25_bge_overlap = []
    bm25_e5_overlap = []
    bge_e5_overlap = []
    keyword_matches = 0
    total_pooled = 0
    
    print("\n=== POOLING OVERLAP ANALYSIS ===")
    for idx, job_row in jobs_dev.iterrows():
        query = str(job_row['Long Description'])
        job_keyword = str(job_row['Primary Keyword']).lower().strip()
        
        bm25_indices = get_bm25_topk(query, tokenized_corpus, bm25, k)
        bge_indices = get_dense_topk(query, bge_corpus, bge_model, k)
        e5_indices = get_dense_topk("query: " + query, e5_corpus, e5_model, k)
        
        bm25_bge_overlap.append(overlap(bm25_indices, bge_indices))
        bm25_e5_overlap.append(overlap(bm25_indices, e5_indices))
        bge_e5_overlap.append(overlap(bge_indices, e5_indices))
        
        # Check keyword matches
        union_indices = set(bm25_indices).union(set(bge_indices)).union(set(e5_indices))
        total_pooled += len(union_indices)
        for c_idx in union_indices:
            if cv_keywords[c_idx] == job_keyword:
                keyword_matches += 1
                
        # Print top-3 for first 3 jobs
        if idx < 3:
            print(f"\n--- Job: {job_row['Position'][:50]}... ---")
            print(f"BM25 Top 3 CVs:")
            for i in bm25_indices[:3]: print(f"  - {cv_texts[i][:100]}...")
            print(f"BGE Top 3 CVs:")
            for i in bge_indices[:3]: print(f"  - {cv_texts[i][:100]}...")
            print(f"E5 Top 3 CVs:")
            for i in e5_indices[:3]: print(f"  - {cv_texts[i][:100]}...")

    print("\n=== PAIRWISE OVERLAP @ 5 ===")
    print(f"BM25 & BGE: {np.mean(bm25_bge_overlap):.2f}")
    print(f"BM25 & E5 : {np.mean(bm25_e5_overlap):.2f}")
    print(f"BGE & E5  : {np.mean(bge_e5_overlap):.2f}")
    print(f"\nPrimary Keyword Match Rate: {keyword_matches}/{total_pooled} ({(keyword_matches/total_pooled)*100:.1f}%)")

if __name__ == "__main__":
    run_pooling_analysis(k=5)
