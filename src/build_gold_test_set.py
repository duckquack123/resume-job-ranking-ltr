import os
import re
import json
import random
import hashlib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def get_5gram_shingles(text):
    if not isinstance(text, str):
        return set()
    words = [w for w in re.split(r'\W+', text.lower()) if w]
    if len(words) < 5:
        return set([' '.join(words)]) if words else set()
    return set([' '.join(words[i:i+5]) for i in range(len(words)-4)])

def jaccard_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def main():
    print("=== Step C: Building Gold Test Set & Annotator Task Exports ===")
    out_dir = "/csehome/m25csa007/m25csa007/rjm-ltr/data/processed"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load 52 eligible test jobs
    print("1. Loading eligible test jobs...")
    jobs_test = pd.read_parquet(os.path.join(out_dir, "jobs_test.parquet"))
    with open(os.path.join(out_dir, "eligible_test_job_ids.txt")) as f:
        eligible_ids = [x.strip() for x in f if x.strip()]
        
    jobs_52 = jobs_test[jobs_test["id"].isin(eligible_ids)].sort_values("id").reset_index(drop=True)
    assert len(jobs_52) == 52, f"Expected 52 jobs, found {len(jobs_52)}"
    print(f"Loaded {len(jobs_52)} eligible test jobs.")
    
    # Verify 134642_job
    job_134642 = jobs_52[jobs_52["id"] == "134642_job"]
    assert len(job_134642) == 1, "134642_job missing from eligible test jobs!"
    print(f"Verified 134642_job: {job_134642['Position'].iloc[0]} at {job_134642['Company Name'].iloc[0]}")
    
    # 2. Load test CV pool and apply excluded_cv_ids.txt
    print("\n2. Loading test CV pool and applying excluded_cv_ids.txt...")
    cvs_test = pd.read_parquet(os.path.join(out_dir, "cvs_test.parquet"))
    print(f"Total test CVs before exclusion: {len(cvs_test)}")
    
    with open(os.path.join(out_dir, "excluded_cv_ids.txt")) as f:
        excluded_cv_ids = set(x.strip() for x in f if x.strip())
    print(f"Excluded CV IDs: {excluded_cv_ids}")
    
    cvs_clean = cvs_test[~cvs_test["id"].isin(excluded_cv_ids)].sort_values("id").reset_index(drop=True)
    print(f"Clean test CVs after exclusion: {len(cvs_clean)}")
    
    cv_ids = cvs_clean["id"].tolist()
    cv_texts = cvs_clean["CV"].fillna("").tolist()
    cv_text_map = dict(zip(cv_ids, cv_texts))
    
    # 3. Build retrievers and encode
    print("\n3. Building BM25 index...")
    tokenized_corpus = [str(doc).lower().split() for doc in cv_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for dense embeddings")
    
    print("Encoding CVs with BGE-small-en-v1.5...")
    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    bge_corpus = bge_model.encode(
        cv_texts, batch_size=256, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True
    )
    
    print("Encoding CVs with E5-small-v2...")
    e5_model = SentenceTransformer("intfloat/e5-small-v2", device=device)
    e5_passages = ["passage: " + str(t) for t in cv_texts]
    e5_corpus = e5_model.encode(
        e5_passages, batch_size=256, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True
    )
    
    # 4. Pooling K=3 per system across all 52 jobs
    print("\n4. Pooling K=3 per system (BM25, BGE, E5) and applying within-pool 5-gram dedup...")
    K = 3
    gold_pool_by_job = {}
    gold_provenance = {}
    total_slots = 0
    total_unique_candidates = 0
    total_dedup_dropped = 0
    
    for _, row in jobs_52.iterrows():
        jid = row["id"]
        jdesc = str(row["Long Description"])
        
        # BM25 top K
        q_tokens = jdesc.lower().split()
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_idx = np.argsort(bm25_scores)[::-1][:K]
        
        # BGE top K
        bge_q = bge_model.encode([jdesc], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        bge_sims = torch.mm(bge_q, bge_corpus.T)[0]
        bge_idx = torch.topk(bge_sims, k=K).indices.cpu().numpy()
        
        # E5 top K
        e5_q = e5_model.encode(["query: " + jdesc], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        e5_sims = torch.mm(e5_q, e5_corpus.T)[0]
        e5_idx = torch.topk(e5_sims, k=K).indices.cpu().numpy()
        
        pool_cvs = {}
        pool_prov = {}
        
        def add_sys(indices, sys_name):
            for i in indices:
                cid = cv_ids[i]
                if cid not in pool_cvs:
                    pool_cvs[cid] = cv_texts[i]
                    pool_prov[cid] = []
                pool_prov[cid].append(sys_name)
                
        add_sys(bm25_idx, "BM25")
        add_sys(bge_idx, "BGE")
        add_sys(e5_idx, "E5")
        
        total_slots += 3 * K
        
        # Within-pool 5-gram dedup (0.70 threshold)
        final_cids = []
        cids_ordered = list(pool_cvs.keys())
        
        for cid in cids_ordered:
            txt = pool_cvs[cid]
            s = get_5gram_shingles(txt)
            is_dupe = False
            for prev_cid in final_cids:
                prev_s = get_5gram_shingles(pool_cvs[prev_cid])
                if jaccard_similarity(s, prev_s) >= 0.70:
                    is_dupe = True
                    break
            if is_dupe:
                total_dedup_dropped += 1
            else:
                final_cids.append(cid)
                
        total_unique_candidates += len(final_cids)
        gold_pool_by_job[jid] = {
            "job_id": jid,
            "job_title": str(row["Position"]),
            "job_desc": jdesc,
            "company": str(row["Company Name"]) if not pd.isna(row["Company Name"]) else "UNKNOWN",
            "role_family": str(row["Role_Family"]),
            "candidate_cv_ids": final_cids
        }
        
        for cid in final_cids:
            pair_id = f"{jid}_{cid}"
            gold_provenance[pair_id] = pool_prov[cid]
            
    print(f"Total retrieval slots (52 * 9): {total_slots}")
    print(f"Total deduplicated candidates across 52 jobs: {total_unique_candidates}")
    print(f"Average candidates per job: {total_unique_candidates / len(jobs_52):.2f}")
    print(f"Within-pool duplicates dropped (Jaccard >= 0.70): {total_dedup_dropped}")
    
    # Save provenance mapping in a separate file (annotators never see this)
    prov_file = os.path.join(out_dir, "gold_provenance.json")
    with open(prov_file, "w") as f:
        json.dump(gold_provenance, f, indent=2)
    with open(prov_file, "rb") as f:
        prov_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Saved {prov_file} (SHA-256: {prov_hash})")
    
    # 5. Stratified Selection of 13 Jobs for Annotator B (seed 42)
    print("\n5. Stratified Selection of 13 Jobs for Annotator B (seed 42)...")
    role_dist = jobs_52["Role_Family"].value_counts(normalize=True)
    target_total = 13
    target_counts = {}
    for role, prop in role_dist.items():
        target_counts[role] = int(round(prop * target_total))
        
    diff = target_total - sum(target_counts.values())
    if diff > 0:
        for role in role_dist.keys():
            target_counts[role] += 1
            diff -= 1
            if diff == 0:
                break
    elif diff < 0:
        for role in role_dist.keys():
            if target_counts[role] > 0:
                target_counts[role] -= 1
                diff += 1
                if diff == 0:
                    break
                    
    print("Target role family allocation for 13 jobs:")
    for r, c in target_counts.items():
        if c > 0:
            print(f"  {r}: {c} / {jobs_52['Role_Family'].value_counts()[r]}")
            
    sampled_b_jobs = []
    grouped = jobs_52.groupby("Role_Family")
    for role, group in grouped:
        n_t = target_counts.get(role, 0)
        if n_t > 0:
            sample_grp = group.sample(n=n_t, random_state=42)
            sampled_b_jobs.append(sample_grp)
            
    df_b = pd.concat(sampled_b_jobs).sort_values("id").reset_index(drop=True)
    annotator_b_jids = df_b["id"].tolist()
    assert len(annotator_b_jids) == 13, f"Expected 13 jobs, got {len(annotator_b_jids)}"
    
    b_ids_file = os.path.join(out_dir, "annotator_b_job_ids.txt")
    with open(b_ids_file, "w") as f:
        for jid in annotator_b_jids:
            f.write(f"{jid}\n")
            
    with open(b_ids_file, "rb") as f:
        b_ids_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Saved {b_ids_file} (SHA-256: {b_ids_hash})")
    print("Annotator B Job IDs:")
    for _, r in df_b.iterrows():
        print(f"  {r['id']}: {r['Position']} ({r['Role_Family']})")
        
    # 6. Export files by whole job
    print("\n6. Exporting annotator task JSON files by whole job...")
    
    def generate_tasks_for_jobs(job_id_list, seed=42):
        rng = random.Random(seed)
        job_tasks = []
        for jid in sorted(job_id_list):
            jdata = gold_pool_by_job[jid]
            cids = list(jdata["candidate_cv_ids"])
            rng.shuffle(cids)
            
            candidates = []
            for cid in cids:
                pair_id = f"{jid}_{cid}"
                candidates.append({
                    "pair_id": pair_id,
                    "cv_id": cid,
                    "cv_text": cv_text_map[cid]
                })
                
            job_tasks.append({
                "job_id": jid,
                "job_title": jdata["job_title"],
                "job_desc": jdata["job_desc"],
                "role_family": jdata["role_family"],
                "candidates": candidates
            })
        return job_tasks
        
    tasks_a = generate_tasks_for_jobs(jobs_52["id"].tolist(), seed=42)
    tasks_b = generate_tasks_for_jobs(annotator_b_jids, seed=42)
    
    file_a = os.path.join(out_dir, "annotator_a_tasks.json")
    with open(file_a, "w") as f:
        json.dump(tasks_a, f, indent=2)
    with open(file_a, "rb") as f:
        hash_a = hashlib.sha256(f.read()).hexdigest()
        
    file_b = os.path.join(out_dir, "annotator_b_tasks.json")
    with open(file_b, "w") as f:
        json.dump(tasks_b, f, indent=2)
    with open(file_b, "rb") as f:
        hash_b = hashlib.sha256(f.read()).hexdigest()
        
    n_pairs_a = sum(len(j["candidates"]) for j in tasks_a)
    n_pairs_b = sum(len(j["candidates"]) for j in tasks_b)
    
    print(f"\nAnnotator A Tasks File: {file_a}")
    print(f"  Jobs: {len(tasks_a)}, Pairs: {n_pairs_a}, SHA-256: {hash_a}")
    print(f"Annotator B Tasks File: {file_b}")
    print(f"  Jobs: {len(tasks_b)}, Pairs: {n_pairs_b}, SHA-256: {hash_b}")
    
    # Save summary manifest
    manifest = {
        "eligible_test_jobs_count": 52,
        "test_cv_pool_clean_count": len(cvs_clean),
        "excluded_cv_ids": list(excluded_cv_ids),
        "systems_pooled": ["BM25", "BGE", "E5"],
        "k_per_system": K,
        "total_pairs_annotator_a": n_pairs_a,
        "total_pairs_annotator_b": n_pairs_b,
        "dedup_dropped_candidates": total_dedup_dropped,
        "hashes": {
            "annotator_b_job_ids": b_ids_hash,
            "gold_provenance": prov_hash,
            "annotator_a_tasks": hash_a,
            "annotator_b_tasks": hash_b
        }
    }
    with open(os.path.join(out_dir, "gold_pool_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nSaved data/processed/gold_pool_manifest.json successfully.")

if __name__ == "__main__":
    main()
