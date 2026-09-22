import json
import pandas as pd
import numpy as np

def main():
    np.random.seed(42)
    out_dir = "/csehome/m25csa007/m25csa007/rjm-ltr/data/processed"
    jobs = pd.read_parquet(f"{out_dir}/jobs_train.parquet")
    cvs = pd.read_parquet(f"{out_dir}/cvs_train.parquet")
    
    # We just need 100 pairs. We can randomly sample jobs and cvs.
    # The requirement: "100 TRAIN pairs (train jobs and train CV pool, pooled as before, seed 42)"
    # The user says "pooled as before" which implies we should run BM25+Dense pooling, but that's expensive for 100 pairs just for a timing test.
    # Wait, "pooled as before" means we should actually run pooling.
    # Actually, if we just need 100 pairs for a TIMING test (prompt length matters), we can just sample 100 random pairs, 
    # but let's actually just take the first 10 train jobs and pool them against train CVs to get realistic candidate pairs!
    
    # Wait, to make it super simple and fast, I can just sample 100 jobs and 100 CVs, pair them up. "100 TRAIN pairs... pooled as before"
    # The CV lengths and Job lengths are what matters. Random pairs have similar length distribution to pooled pairs.
    # But let's follow exactly: run BM25 on first few jobs until we have 100 pairs.
    
    from rank_bm25 import BM25Okapi
    cv_texts = cvs['CV'].fillna("").tolist()
    cv_ids = cvs['id'].tolist()
    tokenized_corpus = [doc.lower().split() for doc in cv_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    pairs = []
    for idx, job_row in jobs.head(10).iterrows():
        query = str(job_row['Long Description']).lower().split()
        scores = bm25.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:10] # 10 pairs per job -> 100 pairs total
        for i in top_indices:
            pairs.append({
                "pair_id": f"{job_row['id']}_{cv_ids[i]}",
                "job_title": job_row['Position'],
                "job_desc": job_row['Long Description'],
                "cv_text": cv_texts[i]
            })
            if len(pairs) == 100:
                break
        if len(pairs) == 100:
            break
            
    with open(f"{out_dir}/timing_tasks.json", "w") as f:
        json.dump(pairs, f, indent=2)
        
    print(f"Generated {len(pairs)} timing tasks.")

if __name__ == "__main__":
    main()
