import pandas as pd
from datasketch import MinHash, MinHashLSH
import re
import numpy as np
from datasets import load_dataset
import random

def get_minhash(text, num_perm, seed=42):
    m = MinHash(num_perm=num_perm, seed=seed)
    if not isinstance(text, str):
        text = ""
    # Use shingle size 3 (trigrams)
    tokens = text.lower().split()
    shingles = [" ".join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    if not shingles:
        shingles = tokens
    for shingle in set(shingles):
        m.update(shingle.encode('utf8'))
    return m

def analyze_dedup_thresholds(df, text_col, num_perm=128):
    # Sample down for speed in analysis
    df = df.sample(n=min(20000, len(df)), random_state=42).reset_index(drop=True)
    
    print(f"Building LSH (N={len(df)}) on '{text_col}' for thresholds [0.7, 0.8, 0.9]...")
    
    hashes = {}
    for idx, row in df.iterrows():
        hashes[idx] = get_minhash(row[text_col], num_perm)
        
    for thresh in [0.7, 0.8, 0.9]:
        lsh = MinHashLSH(threshold=thresh, num_perm=num_perm)
        for idx, m in hashes.items():
            lsh.insert(idx, m)
            
        drop_count = 0
        pairs = []
        seen = set()
        for idx, m in hashes.items():
            if idx in seen: continue
            res = lsh.query(m)
            if len(res) > 1:
                res_set = set(res)
                drop_count += len(res_set) - 1
                seen.update(res_set)
                
                # save some pairs
                res_list = list(res_set)
                if len(pairs) < 5:
                    pairs.append((df.iloc[res_list[0]][text_col], df.iloc[res_list[1]][text_col]))
                    
        print(f"\nThreshold {thresh}: would drop {drop_count} near-duplicates (in 20k sample).")
        if thresh == 0.8:
            print(f"Example Pairs (Threshold {thresh}):")
            for i, (t1, t2) in enumerate(pairs[:3]):
                print(f"--- Pair {i+1} ---")
                print(f"Doc 1: {t1[:200]}...")
                print(f"Doc 2: {t2[:200]}...")

if __name__ == "__main__":
    jobs_ds = load_dataset("lang-uk/recruitment-dataset-job-descriptions-english", split="train")
    jobs_df = jobs_ds.to_pandas().dropna(subset=['Long Description']).reset_index(drop=True)
    jobs_df = jobs_df[jobs_df['Long Description_lang'] == 'en']
    print("\n=== JOBS DEDUP ANALYSIS ===")
    analyze_dedup_thresholds(jobs_df, 'Long Description')
    
    cvs_ds = load_dataset("lang-uk/recruitment-dataset-candidate-profiles-english", split="train")
    cvs_df = cvs_ds.to_pandas().dropna(subset=['CV']).reset_index(drop=True)
    cvs_df = cvs_df[cvs_df['CV_lang'] == 'en']
    print("\n=== CVs DEDUP ANALYSIS ===")
    analyze_dedup_thresholds(cvs_df, 'CV')
