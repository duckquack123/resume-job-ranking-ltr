import json
import numpy as np
import pandas as pd
from collections import defaultdict

def dcg_at_k(r, k=10):
    r = np.asarray(r, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k=10):
    dcg = dcg_at_k(r, k)
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if idcg == 0.0:
        return 1.0 # Or 0.0, but if all items are 0, rank order does not matter
    return dcg / idcg

def mrr_at_k(r, k=10, rel_thresh=2):
    r = np.asarray(r, dtype=float)[:k]
    for idx, rel in enumerate(r):
        if rel >= rel_thresh:
            return 1.0 / (idx + 1)
    return 0.0

def expected_random_ndcg_at_k(r, k=10):
    r = np.asarray(r, dtype=float)
    m = len(r)
    if m == 0:
        return 0.0
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if idcg == 0.0:
        return 1.0
    # Expected gain at any position is the mean gain over all m candidates
    mean_gain = np.mean(2**r - 1)
    k_eff = min(k, m)
    discount_sum = np.sum(1.0 / np.log2(np.arange(2, k_eff + 2)))
    exp_dcg = mean_gain * discount_sum
    return exp_dcg / idcg

def expected_random_mrr(r, k=10, rel_thresh=2):
    r = np.asarray(r, dtype=float)
    m = len(r)
    rel_count = np.sum(r >= rel_thresh)
    if rel_count == 0 or m == 0:
        return 0.0
    # Monte carlo with 10,000 runs for exact expectation
    ranks = []
    np.random.seed(42)
    for _ in range(10000):
        perm = np.random.permutation(r)
        rr = 0.0
        for idx, val in enumerate(perm[:k]):
            if val >= rel_thresh:
                rr = 1.0 / (idx + 1)
                break
        ranks.append(rr)
    return np.mean(ranks)

def main():
    df = pd.read_parquet("data/processed/features_dev_86.parquet")
    print(f"Loaded {len(df)} dev pairs across {df['job_id'].nunique()} jobs.")
    
    jobs = df["job_id"].unique()
    systems = ["bm25", "bge", "e5"]
    
    per_job_results = []
    
    for jid in sorted(jobs):
        df_j = df[df["job_id"] == jid].copy()
        labels = df_j["human_label"].values
        m = len(labels)
        
        row = {"job_id": jid, "num_candidates": m}
        
        # Random baseline
        rand_ndcg = expected_random_ndcg_at_k(labels, k=10)
        rand_mrr2 = expected_random_mrr(labels, k=10, rel_thresh=2)
        rand_mrr1 = expected_random_mrr(labels, k=10, rel_thresh=1)
        row["random_ndcg10"] = rand_ndcg
        row["random_mrr_ge2"] = rand_mrr2
        row["random_mrr_ge1"] = rand_mrr1
        
        for sys_name in systems:
            # Sort descending by system score
            # Secondary tie breaker: random/stable
            df_sorted = df_j.sort_values(by=sys_name, ascending=False)
            sys_labels = df_sorted["human_label"].values
            
            ndcg = ndcg_at_k(sys_labels, k=10)
            mrr2 = mrr_at_k(sys_labels, k=10, rel_thresh=2)
            mrr1 = mrr_at_k(sys_labels, k=10, rel_thresh=1)
            
            row[f"{sys_name}_ndcg10"] = ndcg
            row[f"{sys_name}_mrr_ge2"] = mrr2
            row[f"{sys_name}_mrr_ge1"] = mrr1
            
        per_job_results.append(row)
        
    df_per_job = pd.DataFrame(per_job_results)
    
    print("\n=== Per-Job Retrieval Evaluation (n = 10 jobs, 86 pairs) ===")
    display_cols = ["job_id", "num_candidates"] + [f"{s}_ndcg10" for s in systems] + ["random_ndcg10"]
    print(df_per_job[display_cols].to_string(index=False))
    
    # Compute summary metrics and Bootstrap CIs
    print("\n=== Summary Metrics & 95% Bootstrap CIs ===")
    np.random.seed(42)
    B = 1000
    
    summary = {}
    for metric_prefix in ["ndcg10", "mrr_ge2", "mrr_ge1"]:
        print(f"\n--- Metric: {metric_prefix} ---")
        for sys_name in systems + ["random"]:
            col = f"{sys_name}_{metric_prefix}"
            vals = df_per_job[col].values
            mean_val = np.mean(vals)
            
            # Job-level cluster bootstrap (resample 10 jobs)
            boot_job_means = []
            for _ in range(B):
                boot_idx = np.random.choice(len(vals), size=len(vals), replace=True)
                boot_job_means.append(np.mean(vals[boot_idx]))
            ci_job = (np.percentile(boot_job_means, 2.5), np.percentile(boot_job_means, 97.5))
            
            print(f"{sys_name:<8}: Mean = {mean_val:.4f} | 95% Job-Cluster CI: [{ci_job[0]:.4f}, {ci_job[1]:.4f}] (n = 10 jobs)")
            summary[f"{sys_name}_{metric_prefix}"] = {
                "mean": float(mean_val),
                "ci_job_low": float(ci_job[0]),
                "ci_job_high": float(ci_job[1])
            }

    # Provenance Mapping Analysis
    print("\n=== Provenance Mapping & High-Score Retrieval Analysis ===")
    # Each row in df has 'provenance' list and 'human_label'
    prov_stats = defaultdict(list)
    exclusive_stats = defaultdict(list)
    
    for _, row in df.iterrows():
        lbl = row["human_label"]
        prov = list(row["provenance"])
        for sys_tag in prov:
            prov_stats[sys_tag].append(lbl)
        if len(prov) == 1:
            exclusive_stats[prov[0]].append(lbl)
        elif len(prov) > 1:
            exclusive_stats["Multiple"].append(lbl)
            
    print("\n--- Overall Retrieval by Provenance (All retrieved pairs per system) ---")
    prov_summary = {}
    for sys_tag in ["BM25", "BGE", "E5"]:
        scores = np.array(prov_stats[sys_tag])
        mean_s = np.mean(scores)
        count_total = len(scores)
        c0 = np.sum(scores == 0)
        c1 = np.sum(scores == 1)
        c2 = np.sum(scores == 2)
        c3 = np.sum(scores == 3)
        pct_rel = (c2 + c3) / count_total * 100
        print(f"System: {sys_tag:<6} | Total: {count_total:>2} | Mean Grade: {mean_s:.2f} | Grades [0/1/2/3]: {c0}/{c1}/{c2}/{c3} | Rel (2-3): {c2+c3} ({pct_rel:.1f}%)")
        prov_summary[sys_tag] = {
            "total_retrieved": int(count_total),
            "mean_score": float(mean_s),
            "grade_distribution": {"0": int(c0), "1": int(c1), "2": int(c2), "3": int(c3)},
            "relevant_count": int(c2 + c3),
            "relevant_pct": float(pct_rel)
        }
        
    print("\n--- Grade 3 (Highly Relevant) Candidates Breakdown ---")
    grade3_pairs = df[df["human_label"] == 3]
    print(f"Total Grade 3 candidates: {len(grade3_pairs)}")
    for _, r in grade3_pairs.iterrows():
        print(f"Pair: {r['pair_id']} | Job: {r['job_id']} | Retrieved by: {r['provenance']} | BM25: {r['bm25']:.1f}, BGE: {r['bge']:.3f}, E5: {r['e5']:.3f}")
        
    print("\n--- Grade 2 (Relevant) Candidates Breakdown ---")
    grade2_pairs = df[df["human_label"] == 2]
    print(f"Total Grade 2 candidates: {len(grade2_pairs)}")
    
    # Save full tables and results to json
    results_out = {
        "per_job": per_job_results,
        "summary": summary,
        "provenance": prov_summary
    }
    with open("data/processed/retriever_eval_dev_86.json", "w") as f:
        json.dump(results_out, f, indent=2)
    print("\nSaved retriever evaluation results to data/processed/retriever_eval_dev_86.json")

if __name__ == "__main__":
    main()
