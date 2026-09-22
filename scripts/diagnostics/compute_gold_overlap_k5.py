import os
import json
import re
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def main():
    print("=== Computing Overlap@5 & Overlap@3 for 52 Gold Test Jobs ===", flush=True)
    out_dir = "data/processed"
    
    # 1. Load 52 eligible test jobs
    jobs_test = pd.read_parquet(os.path.join(out_dir, "jobs_test.parquet"))
    with open(os.path.join(out_dir, "eligible_test_job_ids.txt")) as f:
        eligible_ids = [x.strip() for x in f if x.strip()]
    jobs_52 = jobs_test[jobs_test["id"].isin(eligible_ids)].sort_values("id").reset_index(drop=True)
    print(f"Loaded {len(jobs_52)} eligible test jobs.", flush=True)
    
    # 2. Load clean test CV pool
    cvs_test = pd.read_parquet(os.path.join(out_dir, "cvs_test.parquet"))
    with open(os.path.join(out_dir, "excluded_cv_ids.txt")) as f:
        excluded_cv_ids = set(x.strip() for x in f if x.strip())
    cvs_clean = cvs_test[~cvs_test["id"].isin(excluded_cv_ids)].sort_values("id").reset_index(drop=True)
    print(f"Loaded {len(cvs_clean)} clean test CVs.", flush=True)
    
    cv_ids = cvs_clean["id"].tolist()
    cv_texts = cvs_clean["CV"].fillna("").tolist()
    
    # 3. Build BM25
    print("Building BM25 index...", flush=True)
    tokenized_corpus = [str(doc).lower().split() for doc in cv_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device for embeddings: {device}", flush=True)
    
    print("Encoding with BGE...", flush=True)
    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    bge_corpus = bge_model.encode(cv_texts, batch_size=256, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
    
    print("Encoding with E5...", flush=True)
    e5_model = SentenceTransformer("intfloat/e5-small-v2", device=device)
    e5_passages = ["passage: " + str(t) for t in cv_texts]
    e5_corpus = e5_model.encode(e5_passages, batch_size=256, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
    
    # 4. Compute overlaps for K=5 and K=3
    for K in [5, 3]:
        print(f"\n--- Overlap Analysis for K = {K} ---", flush=True)
        bm25_bge_overlaps = []
        bm25_e5_overlaps = []
        bge_e5_overlaps = []
        
        all_union_cids = []
        total_slots = 0
        
        for _, row in jobs_52.iterrows():
            jdesc = str(row["Long Description"])
            
            # BM25 top K
            q_tokens = jdesc.lower().split()
            bm25_scores = bm25.get_scores(q_tokens)
            bm25_top = [cv_ids[i] for i in np.argsort(bm25_scores)[::-1][:K]]
            
            # BGE top K
            bge_q = bge_model.encode([jdesc], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            bge_sims = torch.mm(bge_q, bge_corpus.T)[0]
            bge_top = [cv_ids[i] for i in torch.topk(bge_sims, k=K).indices.cpu().numpy()]
            
            # E5 top K
            e5_q = e5_model.encode(["query: " + jdesc], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            e5_sims = torch.mm(e5_q, e5_corpus.T)[0]
            e5_top = [cv_ids[i] for i in torch.topk(e5_sims, k=K).indices.cpu().numpy()]
            
            # Pairwise intersections
            s_bm25 = set(bm25_top)
            s_bge = set(bge_top)
            s_e5 = set(e5_top)
            
            bm25_bge_overlaps.append(len(s_bm25.intersection(s_bge)))
            bm25_e5_overlaps.append(len(s_bm25.intersection(s_e5)))
            bge_e5_overlaps.append(len(s_bge.intersection(s_e5)))
            
            union_k = s_bm25.union(s_bge).union(s_e5)
            all_union_cids.append(len(union_k))
            total_slots += 3 * K
            
        mean_bm25_bge = np.mean(bm25_bge_overlaps)
        mean_bm25_e5 = np.mean(bm25_e5_overlaps)
        mean_bge_e5 = np.mean(bge_e5_overlaps)
        
        pct_bm25_bge = (mean_bm25_bge / K) * 100
        pct_bm25_e5 = (mean_bm25_e5 / K) * 100
        pct_bge_e5 = (mean_bge_e5 / K) * 100
        
        total_union_size = sum(all_union_cids)
        avg_union_per_job = np.mean(all_union_cids)
        total_multi_overlap = total_slots - total_union_size
        multi_overlap_pct = (total_multi_overlap / total_slots) * 100
        
        print(f"Total Retrieval Slots (52 * {3*K}): {total_slots}")
        print(f"Total Deduplicated Union Candidates: {total_union_size}")
        print(f"Average Candidates per Job: {avg_union_per_job:.2f} / {3*K}")
        print(f"Multi-System Overlap Slots: {total_multi_overlap} ({multi_overlap_pct:.2f}%)")
        print(f"Pairwise Overlaps per Job (count and % of K={K}):")
        print(f"  BM25 & BGE: {mean_bm25_bge:.3f} / {K} ({pct_bm25_bge:.2f}%)")
        print(f"  BM25 & E5 : {mean_bm25_e5:.3f} / {K} ({pct_bm25_e5:.2f}%)")
        print(f"  BGE & E5  : {mean_bge_e5:.3f} / {K} ({pct_bge_e5:.2f}%)")
        
        # Save summary
        res_data = {
            f"K_{K}": {
                "total_slots": int(total_slots),
                "total_union_size": int(total_union_size),
                "avg_union_per_job": float(avg_union_per_job),
                "total_multi_overlap": int(total_multi_overlap),
                "multi_overlap_pct": float(multi_overlap_pct),
                "pairwise": {
                    "bm25_bge_mean": float(mean_bm25_bge),
                    "bm25_bge_pct": float(pct_bm25_bge),
                    "bm25_e5_mean": float(mean_bm25_e5),
                    "bm25_e5_pct": float(pct_bm25_e5),
                    "bge_e5_mean": float(mean_bge_e5),
                    "bge_e5_pct": float(pct_bge_e5)
                }
            }
        }
        
        with open(f"data/processed/gold_overlap_k{K}.json", "w") as f:
            json.dump(res_data, f, indent=2)

if __name__ == "__main__":
    main()
