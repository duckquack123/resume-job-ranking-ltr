"""
cross_split_dedup.py  –  Remove train items near-duplicate to any dev/test item.
Operates ONLY on the TRAIN side; frozen dev/test IDs and hashes are never touched.
Config: word 5-grams, num_perm=128, threshold=0.70 (same as test_pipeline.py).
"""
import pandas as pd
import re
import os
import hashlib
from datasketch import MinHash, MinHashLSH

OUT_DIR = "data/processed"

def get_minhash_5gram(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    tokens = [t for t in re.split(r"\W+", text) if t]
    shingles = [" ".join(tokens[i: i+5]) for i in range(max(1, len(tokens)-4))]
    for s in set(shingles):
        m.update(s.encode("utf-8"))
    return m


def run_cross_split_dedup():
    print("Loading parquets...")
    train_jobs = pd.read_parquet(f"{OUT_DIR}/jobs_train.parquet")
    dev_jobs   = pd.read_parquet(f"{OUT_DIR}/jobs_dev.parquet")
    test_jobs  = pd.read_parquet(f"{OUT_DIR}/jobs_test.parquet")

    train_cvs = pd.read_parquet(f"{OUT_DIR}/cvs_train.parquet")
    dev_cvs   = pd.read_parquet(f"{OUT_DIR}/cvs_dev.parquet")
    test_cvs  = pd.read_parquet(f"{OUT_DIR}/cvs_test.parquet")

    THRESHOLD = 0.70
    NUM_PERM  = 128

    def find_train_ids_to_drop(train_df, eval_dfs, text_col):
        """Build LSH from eval items, query each train item, return train IDs to drop."""
        lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
        eval_hashes = {}
        # Index all eval items
        for split_name, df in eval_dfs.items():
            for local_idx, row in df.iterrows():
                key = (split_name, local_idx)
                h = get_minhash_5gram(row[text_col], NUM_PERM)
                eval_hashes[key] = h
                lsh.insert(key, h)
        print(f"  Indexed {len(eval_hashes)} eval items for col '{text_col}'.")

        to_drop_ids = set()
        pair_counts = {"train_vs_dev": 0, "train_vs_test": 0}
        for local_idx, row in train_df.iterrows():
            h = get_minhash_5gram(row[text_col], NUM_PERM)
            results = lsh.query(h)
            for (split_name, _) in results:
                to_drop_ids.add(row["id"])
                if split_name == "dev":
                    pair_counts["train_vs_dev"] += 1
                elif split_name == "test":
                    pair_counts["train_vs_test"] += 1

        print(f"  Pairs found: {pair_counts}")
        return to_drop_ids

    print("\n=== JOBS: cross-split near-duplicate check ===")
    job_eval = {"dev": dev_jobs, "test": test_jobs}
    job_drop_ids = find_train_ids_to_drop(train_jobs, job_eval, "Long Description")
    
    # Also check dev vs test (report only; don't remove frozen IDs)
    dv_test_lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    for local_idx, row in test_jobs.iterrows():
        dv_test_lsh.insert(local_idx, get_minhash_5gram(row["Long Description"], NUM_PERM))
    dev_test_pairs = 0
    for local_idx, row in dev_jobs.iterrows():
        h = get_minhash_5gram(row["Long Description"], NUM_PERM)
        if dv_test_lsh.query(h):
            dev_test_pairs += 1
    print(f"  Dev vs Test job pairs (no action taken): {dev_test_pairs}")

    print(f"\nJob train items to drop: {len(job_drop_ids)} / {len(train_jobs)}")
    train_jobs_clean = train_jobs[~train_jobs["id"].isin(job_drop_ids)]
    print(f"Job train size after drop: {len(train_jobs_clean)}")

    print("\n=== CVs: cross-split near-duplicate check ===")
    cv_eval = {"dev": dev_cvs, "test": test_cvs}
    cv_drop_ids = find_train_ids_to_drop(train_cvs, cv_eval, "CV")

    # dev vs test CVs
    dv_test_cv_lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    for local_idx, row in test_cvs.iterrows():
        dv_test_cv_lsh.insert(local_idx, get_minhash_5gram(row["CV"], NUM_PERM))
    dev_test_cv_pairs = 0
    for local_idx, row in dev_cvs.iterrows():
        h = get_minhash_5gram(row["CV"], NUM_PERM)
        if dv_test_cv_lsh.query(h):
            dev_test_cv_pairs += 1
    print(f"  Dev vs Test CV pairs (no action taken): {dev_test_cv_pairs}")

    print(f"\nCV train items to drop: {len(cv_drop_ids)} / {len(train_cvs)}")
    train_cvs_clean = train_cvs[~train_cvs["id"].isin(cv_drop_ids)]
    print(f"CV train size after drop: {len(train_cvs_clean)}")

    if len(job_drop_ids) > 0 or len(cv_drop_ids) > 0:
        print("\nSaving cleaned train files...")
        train_jobs_clean.to_parquet(f"{OUT_DIR}/jobs_train.parquet", index=False)
        train_cvs_clean.to_parquet(f"{OUT_DIR}/cvs_train.parquet", index=False)

        # Re-hash and report
        with open(f"{OUT_DIR}/jobs_train.parquet", "rb") as f:
            print(f"jobs_train.parquet SHA-256 (after drop): {hashlib.sha256(f.read()).hexdigest()}")
        with open(f"{OUT_DIR}/cvs_train.parquet", "rb") as f:
            print(f"cvs_train.parquet SHA-256 (after drop): {hashlib.sha256(f.read()).hexdigest()}")
    else:
        print("\nNo items to drop; train files unchanged.")

    print("\n=== Summary ===")
    print(f"Jobs train: {len(train_jobs)} -> {len(train_jobs_clean)} (removed {len(job_drop_ids)})")
    print(f"CVs  train: {len(train_cvs)}  -> {len(train_cvs_clean)}  (removed {len(cv_drop_ids)})")
    print(f"Dev  jobs:  {len(dev_jobs)} (frozen, unchanged)")
    print(f"Test jobs:  {len(test_jobs)} (frozen, unchanged)")
    print(f"Dev  CVs:   {len(dev_cvs)} (frozen, unchanged)")
    print(f"Test CVs:   {len(test_cvs)} (frozen, unchanged)")

if __name__ == "__main__":
    run_cross_split_dedup()
