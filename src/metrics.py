import pandas as pd
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
import re
import os
import hashlib
import json

OUT_DIR = "data/processed"

def get_minhash(text, num_perm=64):
    m = MinHash(num_perm=num_perm)
    if not isinstance(text, str): text = ""
    tokens = re.split(r'\W+', text.lower())
    for token in set(tokens):
        if token: m.update(token.encode('utf8'))
    return m

def run_metrics():
    # 7. Frozen IDs hashes and counts
    with open(f"{OUT_DIR}/test_job_ids.txt", "rb") as f: test_ids_bytes = f.read()
    with open(f"{OUT_DIR}/dev_job_ids.txt", "rb") as f: dev_ids_bytes = f.read()
    test_ids = test_ids_bytes.decode('utf-8').strip().split('\n')
    dev_ids = dev_ids_bytes.decode('utf-8').strip().split('\n')
    print("=== FROZEN IDs ===")
    print(f"Test Count: {len(test_ids)}, SHA-256: {hashlib.sha256(test_ids_bytes).hexdigest()}")
    print(f"Dev Count: {len(dev_ids)}, SHA-256: {hashlib.sha256(dev_ids_bytes).hexdigest()}")
    
    # 6. Distributions and Counts
    print("\n=== SPLIT SIZES & DISTRIBUTIONS ===")
    train = pd.read_parquet(f"{OUT_DIR}/jobs_train.parquet")
    dev = pd.read_parquet(f"{OUT_DIR}/jobs_dev.parquet")
    test = pd.read_parquet(f"{OUT_DIR}/jobs_test.parquet")
    
    cv_train = pd.read_parquet(f"{OUT_DIR}/cvs_train.parquet")
    cv_dev = pd.read_parquet(f"{OUT_DIR}/cvs_dev.parquet")
    cv_test = pd.read_parquet(f"{OUT_DIR}/cvs_test.parquet")
    
    print(f"CV Pools -> Train: {len(cv_train)}, Dev: {len(cv_dev)}, Test: {len(cv_test)}")
    
    for name, df in [("Train", train), ("Dev", dev), ("Test", test)]:
        print(f"\n[{name} Jobs: {len(df)}]")
        print("Role Family Distribution:")
        print(df['Role_Family'].value_counts().to_string())
        print("Top 10 Companies:")
        print(df['Company Name'].value_counts().head(10).to_string())
        
    print("\nGold Job Count per Role Family (Test Split):")
    print(test['Role_Family'].value_counts().to_string())

    # 5. Full data dedup counts at 0.7, 0.8, 0.9
    print("\n=== FULL DATA DEDUP COUNTS ===")
    # Load raw jobs and CVs for deduplication tests
    print("Loading raw datasets for dedup tests...")
    jobs_ds = load_dataset("lang-uk/recruitment-dataset-job-descriptions-english", split="train")
    jobs = jobs_ds.to_pandas().dropna(subset=['Long Description']).reset_index(drop=True)
    jobs = jobs.loc[:, ~jobs.columns.duplicated()]
    jobs = jobs[jobs['Long Description_lang'] == 'en']
    
    cvs_ds = load_dataset("lang-uk/recruitment-dataset-candidate-profiles-english", split="train")
    cvs = cvs_ds.to_pandas().dropna(subset=['CV']).reset_index(drop=True)
    cvs = cvs.loc[:, ~cvs.columns.duplicated()]
    cvs = cvs[cvs['CV_lang'] == 'en']
    
    print(f"Total raw jobs (EN): {len(jobs)}")
    print(f"Total raw CVs (EN): {len(cvs)}")
    
    def test_thresholds(df, col, thresholds=[0.7, 0.8, 0.9], num_perm=64):
        # Compute hashes once
        print("Hashing...")
        hashes = {idx: get_minhash(row[col], num_perm=num_perm) for idx, row in df.iterrows()}
        
        counts = {}
        for thr in thresholds:
            print(f"Testing threshold {thr}...")
            lsh = MinHashLSH(threshold=thr, num_perm=num_perm)
            for idx, m in hashes.items(): lsh.insert(idx, m)
            to_drop = set()
            for idx, m in hashes.items():
                if idx in to_drop: continue
                res = set(lsh.query(m))
                res.remove(idx)
                to_drop.update(res)
            counts[thr] = len(to_drop)
        return counts

    print("\nJobs Dedup Drops:")
    job_drops = test_thresholds(jobs, 'Long Description')
    for thr, count in job_drops.items():
        print(f"  Thr {thr}: Dropped {count} (Kept {len(jobs)-count})")
        
    print("\nCVs Dedup Drops:")
    # CVs is 210k rows, hashing 210k docs takes a few mins
    cv_drops = test_thresholds(cvs, 'CV')
    for thr, count in cv_drops.items():
        print(f"  Thr {thr}: Dropped {count} (Kept {len(cvs)-count})")

if __name__ == "__main__":
    run_metrics()
