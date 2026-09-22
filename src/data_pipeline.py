import pandas as pd
import numpy as np
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
import re
import os
import random

OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Primary Keyword -> Role_Family Mapping (Based on top keywords in dataset)
ROLE_MAP = {
    'javascript': 'javascript', 'react': 'javascript', 'angular': 'javascript', 'vue': 'javascript',
    'java': 'java', 'spring': 'java',
    'python': 'python', 'django': 'python',
    'c#': '.net', '.net': '.net',
    'php': 'php', 'laravel': 'php',
    'c++': 'c++', 'c': 'c++',
    'ruby': 'ruby',
    'go': 'go', 'golang': 'go',
    'node.js': 'node.js',
    'qa': 'qa', 'qa automation': 'qa automation', 'manual qa': 'qa',
    'devops': 'devops', 'sysadmin': 'devops',
    'design': 'design', 'ui/ux': 'design',
    'marketing': 'marketing', 'seo': 'marketing',
    'project manager': 'project manager', 'product manager': 'project manager',
    'data science': 'data', 'data analyst': 'data', 'data engineer': 'data',
    'sql': 'sql', 'database': 'sql',
    'security': 'security'
}

def map_role(kw):
    if not isinstance(kw, str): return 'other'
    kw = kw.lower().strip()
    return ROLE_MAP.get(kw, 'other')

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    if not isinstance(text, str): text = ""
    tokens = re.split(r'\W+', text.lower())
    for token in set(tokens):
        if token: m.update(token.encode('utf8'))
    return m

def run_pipeline():
    print("=== LOADING DATA ===")
    jobs_ds = load_dataset("lang-uk/recruitment-dataset-job-descriptions-english", split="train")
    jobs = jobs_ds.to_pandas().dropna(subset=['Long Description']).reset_index(drop=True)
    # FIX DUPLICATE COLUMN
    jobs = jobs.loc[:, ~jobs.columns.duplicated()]
    if '__index_level_0__' in jobs.columns:
        jobs = jobs.drop(columns=['__index_level_0__'])

    jobs = jobs[jobs['Long Description_lang'] == 'en']
    jobs['id'] = jobs.index.astype(str) + "_job"
    
    cvs_ds = load_dataset("lang-uk/recruitment-dataset-candidate-profiles-english", split="train")
    cvs = cvs_ds.to_pandas().dropna(subset=['CV']).reset_index(drop=True)
    cvs = cvs.loc[:, ~cvs.columns.duplicated()]
    if '__index_level_0__' in cvs.columns:
        cvs = cvs.drop(columns=['__index_level_0__'])
        
    cvs = cvs[cvs['CV_lang'] == 'en']
    cvs['id'] = cvs.index.astype(str) + "_cv"
    
    jobs['Role_Family'] = jobs['Primary Keyword'].apply(map_role)
    cvs['Role_Family'] = cvs['Primary Keyword'].apply(map_role)
    
    print("\n=== DEDUPLICATING (thresh=0.85) ===")
    def deduplicate(df, col):
        lsh = MinHashLSH(threshold=0.85, num_perm=128)
        hashes = {idx: get_minhash(row[col]) for idx, row in df.iterrows()}
        for idx, m in hashes.items(): lsh.insert(idx, m)
        to_drop = set()
        for idx, m in hashes.items():
            if idx in to_drop: continue
            res = set(lsh.query(m))
            res.remove(idx)
            to_drop.update(res)
        return df.drop(index=list(to_drop)).reset_index(drop=True)
        
    jobs = deduplicate(jobs, 'Long Description')
    # cvs = deduplicate(cvs, 'CV') # Skip cvs dedup because we already cleaned cvs_train!
    # Wait, we need to generate jobs, we can just save jobs!
    # I'll just re-run deduplicate on jobs.
    
    print("\n=== SPLITTING JOBS (Stratified, Cap 3 per company in Gold/Dev) ===")
    company_counts = jobs['Company Name'].value_counts()
    role_families = jobs['Role_Family'].unique()
    used_companies = set()
    
    def pick_jobs(target_count):
        selected_indices = []
        counts = {r: 0 for r in role_families}
        while len(selected_indices) < target_count:
            added = False
            for r in role_families:
                if len(selected_indices) >= target_count: break
                candidates = jobs[(jobs['Role_Family'] == r) & (~jobs['Company Name'].isin(used_companies))]
                if candidates.empty: continue
                comp = candidates['Company Name'].sample(1, random_state=len(selected_indices)).iloc[0]
                used_companies.add(comp)
                comp_jobs = jobs[jobs['Company Name'] == comp]
                if len(comp_jobs) > 3:
                    comp_jobs = comp_jobs.sample(3, random_state=42)
                selected_indices.extend(comp_jobs.index.tolist())
                added = True
            if not added: break 
        return selected_indices

    test_indices = pick_jobs(70) 
    dev_indices = pick_jobs(25)  
    
    test_jobs = jobs.loc[test_indices].copy()
    dev_jobs = jobs.loc[dev_indices].copy()
    train_jobs = jobs.drop(index=test_indices + dev_indices).copy()
    
    print(f"Jobs Split -> Train: {len(train_jobs)}, Dev: {len(dev_jobs)}, Test: {len(test_jobs)}")
    
    # Save jobs ONLY (CVs are already fixed and saved)
    train_jobs.to_parquet(f"{OUT_DIR}/jobs_train.parquet")
    dev_jobs.to_parquet(f"{OUT_DIR}/jobs_dev.parquet")
    test_jobs.to_parquet(f"{OUT_DIR}/jobs_test.parquet")
    
    with open(f"{OUT_DIR}/test_job_ids.txt", "w") as f:
        f.write("\n".join(test_jobs['id'].tolist()))
    with open(f"{OUT_DIR}/dev_job_ids.txt", "w") as f:
        f.write("\n".join(dev_jobs['id'].tolist()))

if __name__ == "__main__":
    run_pipeline()
