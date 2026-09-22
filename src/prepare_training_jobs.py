import os
import json
import hashlib
import random
import re
import pandas as pd
from collections import defaultdict

BLOCKLIST = {
    'hr', 'sales', 'support', 'recruiter', 'business analyst', 
    'lead generation', 'scrum master', 'salesforce', 'artist', 
    'legal', 'finance', 'accounting', 'copywriter', 'content', 
    'invoicing', 'engagement manager', 'growth', 'business development', 
    'delivery manager', 'vfx', 'project manager', 'marketing', 'product manager', 'design'
}

ALLOWED_OTHER = {'android', 'ios', 'unity', 'flutter', 'scala', 'rust'}

def is_eligible(row):
    rf = str(row['Role_Family']).lower().strip()
    kw = str(row['Primary Keyword']).lower().strip()
    if rf in BLOCKLIST or rf in ['marketing', 'design', 'project manager']:
        return False
    if kw in BLOCKLIST:
        return False
    if rf == 'other':
        return kw in ALLOWED_OTHER
    if kw == 'other':
        return False
    return True

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
    random.seed(42)
    
    # Load train jobs
    jobs_df = pd.read_parquet("data/processed/jobs_train.parquet")
    print(f"Total jobs in jobs_train.parquet: {len(jobs_df)}")
    
    # Filter eligible
    jobs_df["is_eligible"] = jobs_df.apply(is_eligible, axis=1)
    eligible_jobs = jobs_df[jobs_df["is_eligible"]].copy()
    print(f"Eligible jobs: {len(eligible_jobs)}")
    
    # Shuffle with seed 42
    eligible_jobs = eligible_jobs.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Cap at 3 per company
    company_counts = defaultdict(int)
    capped_jobs = []
    
    for _, row in eligible_jobs.iterrows():
        comp = row["Company Name"] if not pd.isna(row["Company Name"]) else "UNKNOWN"
        if company_counts[comp] < 3:
            company_counts[comp] += 1
            capped_jobs.append(row)
            
    capped_df = pd.DataFrame(capped_jobs)
    print(f"Jobs after capping at max 3 per company: {len(capped_df)}")
    
    # Stratified sampling by Role_Family to exactly 500 jobs
    target_total = 500
    role_dist = capped_df["Role_Family"].value_counts(normalize=True)
    
    target_counts = {}
    for role, prop in role_dist.items():
        target_counts[role] = int(round(prop * target_total))
        
    diff = target_total - sum(target_counts.values())
    if diff > 0:
        for role in role_dist.keys():
            target_counts[role] += 1
            diff -= 1
            if diff == 0: break
    elif diff < 0:
        for role in role_dist.keys():
            if target_counts[role] > 0:
                target_counts[role] -= 1
                diff += 1
                if diff == 0: break
                
    print("\nTarget counts per role family:")
    for r, c in target_counts.items():
        print(f"  {r}: {c}")
        
    # Group and sample, applying near-deduplication on the fly
    sampled_jobs = []
    selected_shingles = []
    
    # Group by role family
    grouped = capped_df.groupby("Role_Family")
    
    for role, group in grouped:
        n_target = target_counts[role]
        group_shuffled = group.sample(frac=1, random_state=42).reset_index(drop=True)
        
        accepted_for_role = []
        for _, row in group_shuffled.iterrows():
            desc = str(row["Long Description"])
            shingles = get_5gram_shingles(desc)
            
            # Check near duplicate against already selected jobs
            is_dupe = False
            for prev_shingles in selected_shingles:
                if jaccard_similarity(shingles, prev_shingles) >= 0.70:
                    is_dupe = True
                    break
            if not is_dupe:
                accepted_for_role.append(row)
                selected_shingles.append(shingles)
                if len(accepted_for_role) == n_target:
                    break
        sampled_jobs.append(pd.DataFrame(accepted_for_role))
        
    sampled_df = pd.concat(sampled_jobs).reset_index(drop=True)
    print(f"\nSampled jobs count after dedup: {len(sampled_df)}")
    
    # Sort IDs and freeze
    job_ids = sorted(sampled_df["id"].tolist())
    out_file = "data/processed/train_500_job_ids.txt"
    with open(out_file, "w") as f:
        for jid in job_ids:
            f.write(f"{jid}\n")
            
    with open(out_file, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
        
    print(f"Frozen ID list: {out_file}")
    print(f"SHA-256: {file_hash}")
    
    # Verify final role family distribution
    print("\n--- Final Role-Family Distribution ---")
    final_dist = sampled_df["Role_Family"].value_counts()
    for role, count in final_dist.items():
        pct = count / len(sampled_df) * 100
        print(f"  {role}: {count} ({pct:.1f}%)")
        
    # Verify company stats
    num_unique_companies = sampled_df["Company Name"].nunique()
    max_per_comp = sampled_df["Company Name"].value_counts().max()
    print(f"\nUnique companies: {num_unique_companies}, Max per company: {max_per_comp}")
    
    # Create 5-fold CV splits grouped by company (seed 42)
    random.seed(42)
    companies = sampled_df["Company Name"].fillna("UNKNOWN").unique()
    random.shuffle(companies)
    
    folds = [[] for _ in range(5)]
    for i, c in enumerate(companies):
        folds[i % 5].append(c)
        
    cv_splits = {}
    for fold_idx in range(5):
        test_comps = set(folds[fold_idx])
        test_ids = sampled_df[sampled_df["Company Name"].fillna("UNKNOWN").isin(test_comps)]["id"].tolist()
        cv_splits[f"fold_{fold_idx}"] = sorted(test_ids)
        
    cv_out = "data/processed/train_500_cv_folds.json"
    with open(cv_out, "w") as f:
        json.dump(cv_splits, f, indent=2)
        
    with open(cv_out, "rb") as f:
        folds_hash = hashlib.sha256(f.read()).hexdigest()
        
    print(f"Created 5-fold CV splits: {cv_out}")
    print(f"SHA-256: {folds_hash}")
    for k, v in cv_splits.items():
        print(f"  {k}: {len(v)} jobs")

if __name__ == "__main__":
    main()
