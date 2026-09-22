import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import lightgbm as lgb
from collections import defaultdict

warnings.filterwarnings("ignore")

def dcg_at_k(r, k=10):
    r = np.asarray(r, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k=10):
    dcg = dcg_at_k(r, k)
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if idcg == 0.0:
        return 1.0
    return dcg / idcg

def expected_random_ndcg_at_k(r, k=10):
    r = np.asarray(r, dtype=float)
    m = len(r)
    if m == 0:
        return 0.0
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if idcg == 0.0:
        return 1.0
    mean_gain = np.mean(2**r - 1)
    k_eff = min(k, m)
    discount_sum = np.sum(1.0 / np.log2(np.arange(2, k_eff + 2)))
    return (mean_gain * discount_sum) / idcg

def compute_scale_free_features(df, group_col='job_id'):
    """
    Computes within-job scale-free features:
    1. zscore: (x - mean) / (std + 1e-6)
    2. ratio_top: x / (max + 1e-6)
    3. rank: rankdata(-x, method='min')
    """
    signals = ['bm25', 'bge', 'e5', 'cv_token_len', 'tech_overlap']
    df_out = df.copy()
    
    for sig in signals:
        z_col = f"{sig}_zscore"
        r_col = f"{sig}_ratio_top"
        rk_col = f"{sig}_rank"
        
        # Z-score within job
        df_out[z_col] = df_out.groupby(group_col)[sig].transform(
            lambda s: (s.values - s.mean()) / (s.std(ddof=0) + 1e-6) if s.std(ddof=0) > 1e-6 else np.zeros(len(s))
        )
        # Ratio to top within job
        df_out[r_col] = df_out.groupby(group_col)[sig].transform(
            lambda s: s.values / (s.max() + 1e-6) if s.max() > 0 else np.zeros(len(s))
        )
        # Rank within job (1 = top)
        df_out[rk_col] = df_out.groupby(group_col)[sig].transform(
            lambda s: rankdata(-s.values, method='min')
        )
    return df_out

def evaluate_predictions_per_job(df_eval, score_col, label_col='human_label', k=10):
    per_job = {}
    for jid, grp in df_eval.groupby("job_id"):
        grp_sorted = grp.sort_values(by=score_col, ascending=False)
        labels = grp_sorted[label_col].values
        per_job[jid] = {
            "ndcg10": ndcg_at_k(labels, k=k),
            "random_ndcg10": expected_random_ndcg_at_k(labels, k=k),
            "labels": labels.tolist()
        }
    return per_job

def run_lambdarank(train_df, dev_df, feature_cols, folds_dict, target_col="judge_label"):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [10],
        'learning_rate': 0.05,
        'num_leaves': 15,
        'min_child_samples': 10,
        'random_state': 42,
        'n_jobs': 4,
        'verbose': -1,
        'n_estimators': 100
    }
    
    # 5-fold CV
    fold_job_ndcgs = []
    fold_rands = []
    
    for fold_name, test_jids in folds_dict.items():
        test_mask = train_df["job_id"].isin(test_jids)
        tr_mask = ~test_mask
        
        df_tr = train_df[tr_mask].sort_values("job_id")
        df_te = train_df[test_mask].sort_values("job_id")
        if len(df_te) == 0:
            continue
            
        X_tr = df_tr[feature_cols]
        y_tr = df_tr[target_col].values
        groups_tr = df_tr.groupby("job_id", sort=False).size().values
        
        X_te = df_te[feature_cols]
        y_te = df_te[target_col].values
        groups_te = df_te.groupby("job_id", sort=False).size().values
        
        model = lgb.LGBMRanker(**params)
        model.fit(
            X_tr, y_tr, group=groups_tr,
            eval_set=[(X_te, y_te)], eval_group=[groups_te],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        
        preds_te = model.predict(X_te)
        df_te_eval = df_te.copy()
        df_te_eval["pred"] = preds_te
        
        pj = evaluate_predictions_per_job(df_te_eval, "pred", label_col=target_col)
        for jid, m in pj.items():
            fold_job_ndcgs.append(m["ndcg10"])
            fold_rands.append(m["random_ndcg10"])
            
    cv_mean = float(np.mean(fold_job_ndcgs))
    cv_rand = float(np.mean(fold_rands))
    
    # Full train
    df_all_tr = train_df.sort_values("job_id")
    X_all_tr = df_all_tr[feature_cols]
    y_all_tr = df_all_tr[target_col].values
    groups_all_tr = df_all_tr.groupby("job_id", sort=False).size().values
    
    full_model = lgb.LGBMRanker(**params)
    full_model.fit(X_all_tr, y_all_tr, group=groups_all_tr)
    
    # Dev predictions
    dev_df_eval = dev_df.copy()
    dev_df_eval["pred"] = full_model.predict(dev_df[feature_cols])
    
    pj_dev = evaluate_predictions_per_job(dev_df_eval, "pred", label_col="human_label")
    
    tuning_jids = dev_df[dev_df["split"] == "tuning"]["job_id"].unique()
    heldout_jids = dev_df[dev_df["split"] == "heldout"]["job_id"].unique()
    
    tuning_ndcgs = [pj_dev[j]["ndcg10"] for j in tuning_jids]
    tuning_rands = [pj_dev[j]["random_ndcg10"] for j in tuning_jids]
    
    heldout_ndcgs = [pj_dev[j]["ndcg10"] for j in heldout_jids]
    heldout_rands = [pj_dev[j]["random_ndcg10"] for j in heldout_jids]
    
    all_ndcgs = [pj_dev[j]["ndcg10"] for j in pj_dev]
    all_rands = [pj_dev[j]["random_ndcg10"] for j in pj_dev]
    
    return {
        "cv_ndcg10_mean": cv_mean,
        "cv_random_mean": cv_rand,
        "cv_job_ndcgs": fold_job_ndcgs,
        "tuning_ndcg10": float(np.mean(tuning_ndcgs)),
        "tuning_rand": float(np.mean(tuning_rands)),
        "heldout_ndcg10": float(np.mean(heldout_ndcgs)),
        "heldout_rand": float(np.mean(heldout_rands)),
        "all_dev_ndcg10": float(np.mean(all_ndcgs)),
        "all_dev_rand": float(np.mean(all_rands)),
        "per_job_dev": pj_dev,
        "feature_importances": dict(zip(feature_cols, [float(x) for x in full_model.feature_importances_]))
    }

def main():
    print("=== Step 1 Post-Review Execution & Revisions ===", flush=True)
    
    # 1. Load judge results
    judge_results_file = "data/processed/train_judge_results.jsonl"
    judge_map = {}
    with open(judge_results_file) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            judge_map[item["pair_id"]] = {
                "judge_label": item["avg_round"],
                "judge_expected_score": item["avg_expected_score"]
            }
    
    # 2. Load train and dev features
    df_train_raw = pd.read_parquet("data/processed/features_train.parquet")
    df_train_raw["pair_id"] = df_train_raw["job_id"] + "_" + df_train_raw["cv_id"]
    df_train_raw["judge_label"] = df_train_raw["pair_id"].map(lambda pid: judge_map[pid]["judge_label"])
    df_train_raw["judge_expected_score"] = df_train_raw["pair_id"].map(lambda pid: judge_map[pid]["judge_expected_score"])
    
    df_dev_raw = pd.read_parquet("data/processed/features_dev_86.parquet")
    
    # 3. Compute scale-free within-job normalized features (z-score, ratio to top, rank)
    print("Computing scale-free within-job normalized features (zscore, ratio_top, rank)...", flush=True)
    train_df = compute_scale_free_features(df_train_raw, group_col='job_id')
    dev_df = compute_scale_free_features(df_dev_raw, group_col='job_id')
    
    # Quantile-binned expected scores for sensitivity analysis (4 bins)
    train_df["judge_quantile_label"] = pd.qcut(train_df["judge_expected_score"], q=4, labels=[0, 1, 2, 3]).astype(int)
    
    # Save updated feature sets
    train_df.to_parquet("data/processed/features_train_scale_free.parquet", index=False)
    dev_df.to_parquet("data/processed/features_dev_86_scale_free.parquet", index=False)
    
    # 4. ITEM 1: Print feature means and stds for Train vs Dev
    exact_features = [
        "bm25_zscore", "bm25_ratio_top", "bm25_rank",
        "bge_zscore", "bge_ratio_top", "bge_rank",
        "e5_zscore", "e5_ratio_top", "e5_rank",
        "cv_token_len_zscore", "cv_token_len_ratio_top", "cv_token_len_rank",
        "tech_overlap_zscore", "tech_overlap_ratio_top", "tech_overlap_rank"
    ]
    
    feat_stats = []
    print("\n=== ITEM 1: Exact Feature List & Split Statistics (Train vs Dev) ===", flush=True)
    print(f"{'Feature':<25} | {'Train Mean':<10} | {'Train Std':<10} | {'Dev Mean':<10} | {'Dev Std':<10}")
    print("-" * 75)
    for feat in exact_features:
        tr_m, tr_s = train_df[feat].mean(), train_df[feat].std()
        dv_m, dv_s = dev_df[feat].mean(), dev_df[feat].std()
        print(f"{feat:<25} | {tr_m:>10.4f} | {tr_s:>10.4f} | {dv_m:>10.4f} | {dv_s:>10.4f}")
        feat_stats.append({
            "feature": feat,
            "train_mean": float(tr_m), "train_std": float(tr_s),
            "dev_mean": float(dv_m), "dev_std": float(dv_s)
        })
        
    # Load folds
    with open("data/processed/train_500_cv_folds.json") as f:
        folds = json.load(f)
    with open("data/processed/excluded_train_job_ids_v2.txt") as f:
        excluded = set(x.strip() for x in f if x.strip())
    clean_folds = {k: [j for j in v if j not in excluded] for k, v in folds.items()}
    
    # 5. ITEM 2: Ablations on judge labels and 86 human pairs
    print("\n=== ITEM 2: Specific Ablations on CV Length & Tech Overlap ===", flush=True)
    ablation_configs = {
        "Full Model (All 15)": exact_features,
        "Drop CV Length": [f for f in exact_features if "cv_token_len" not in f],
        "Drop Tech Overlap": [f for f in exact_features if "tech_overlap" not in f],
        "Drop Both (Lex+Dense)": [f for f in exact_features if "cv_token_len" not in f and "tech_overlap" not in f],
        "CV Length Alone": [f for f in exact_features if "cv_token_len" in f],
        "Tech Overlap Alone": [f for f in exact_features if "tech_overlap" in f]
    }
    
    ablation_res = {}
    print(f"{'Configuration':<25} | {'CV (Judge)':<10} | {'Tuning (45)':<11} | {'Heldout (41)':<12} | {'All Dev (86)':<12}")
    print("-" * 80)
    for name, cols in ablation_configs.items():
        res = run_lambdarank(train_df, dev_df, cols, clean_folds, target_col="judge_label")
        ablation_res[name] = res
        print(f"{name:<25} | {res['cv_ndcg10_mean']:.4f}     | {res['tuning_ndcg10']:.4f}      | {res['heldout_ndcg10']:.4f}       | {res['all_dev_ndcg10']:.4f}")
        
    # CV length Spearman correlations
    tr_sp, tr_sp_p = spearmanr(train_df["cv_token_len"], train_df["judge_label"])
    dv_sp, dv_sp_p = spearmanr(dev_df["cv_token_len"], dev_df["human_label"])
    print(f"\nCV Length Spearman with Judge on Train (n=4283): {tr_sp:.4f} (p={tr_sp_p:.2e})")
    print(f"CV Length Spearman with Human on Dev (n=86):     {dv_sp:.4f} (p={dv_sp_p:.2e})")

    # 6. ITEM 3: Paired Bootstrap over 10 Jobs (LambdaMART vs BGE vs Random)
    print("\n=== ITEM 3: Paired Bootstrap over 10 Jobs on Dev ===", flush=True)
    full_res = ablation_res["Full Model (All 15)"]
    lm_pj = full_res["per_job_dev"]
    
    # BGE scores per job on dev
    bge_pj = evaluate_predictions_per_job(dev_df, "bge", label_col="human_label")
    
    job_ids_dev = sorted(list(lm_pj.keys()))
    comparison_table = []
    
    for jid in job_ids_dev:
        split = dev_df[dev_df["job_id"] == jid]["split"].iloc[0]
        lm_score = lm_pj[jid]["ndcg10"]
        bge_score = bge_pj[jid]["ndcg10"]
        rand_score = lm_pj[jid]["random_ndcg10"]
        comparison_table.append({
            "job_id": jid,
            "split": split,
            "lambdamart": lm_score,
            "bge": bge_score,
            "random": rand_score,
            "diff_lm_bge": lm_score - bge_score,
            "diff_lm_rand": lm_score - rand_score
        })
        
    df_comp = pd.DataFrame(comparison_table)
    print(df_comp[["job_id", "split", "lambdamart", "bge", "random", "diff_lm_bge", "diff_lm_rand"]].to_string(index=False))
    
    # Paired Bootstrap
    np.random.seed(42)
    B = 1000
    boot_lm_minus_bge = []
    boot_lm_minus_rand = []
    boot_bge_minus_rand = []
    
    diff_lm_bge = df_comp["diff_lm_bge"].values
    diff_lm_rand = df_comp["diff_lm_rand"].values
    diff_bge_rand = (df_comp["bge"] - df_comp["random"]).values
    
    for _ in range(B):
        idx = np.random.choice(len(job_ids_dev), size=len(job_ids_dev), replace=True)
        boot_lm_minus_bge.append(np.mean(diff_lm_bge[idx]))
        boot_lm_minus_rand.append(np.mean(diff_lm_rand[idx]))
        boot_bge_minus_rand.append(np.mean(diff_bge_rand[idx]))
        
    ci_lm_bge = (np.percentile(boot_lm_minus_bge, 2.5), np.percentile(boot_lm_minus_bge, 97.5))
    ci_lm_rand = (np.percentile(boot_lm_minus_rand, 2.5), np.percentile(boot_lm_minus_rand, 97.5))
    ci_bge_rand = (np.percentile(boot_bge_minus_rand, 2.5), np.percentile(boot_bge_minus_rand, 97.5))
    
    print(f"\nLambdaMART Mean: {df_comp['lambdamart'].mean():.4f}")
    print(f"BGE Mean:        {df_comp['bge'].mean():.4f}")
    print(f"Random Mean:     {df_comp['random'].mean():.4f}")
    print(f"Paired Diff (LambdaMART - BGE):  {np.mean(diff_lm_bge):+.4f} | 95% CI: [{ci_lm_bge[0]:+.4f}, {ci_lm_bge[1]:+.4f}]")
    print(f"Paired Diff (LambdaMART - Rand): {np.mean(diff_lm_rand):+.4f} | 95% CI: [{ci_lm_rand[0]:+.4f}, {ci_lm_rand[1]:+.4f}]")
    print(f"Paired Diff (BGE - Rand):        {np.mean(diff_bge_rand):+.4f} | 95% CI: [{ci_bge_rand[0]:+.4f}, {ci_bge_rand[1]:+.4f}]")

    # Company-cluster bootstrap for 5-fold CV numbers
    cv_job_scores = np.array(full_res["cv_job_ndcgs"])
    boot_cv_means = [np.mean(np.random.choice(cv_job_scores, size=len(cv_job_scores), replace=True)) for _ in range(B)]
    ci_cv = (np.percentile(boot_cv_means, 2.5), np.percentile(boot_cv_means, 97.5))
    print(f"\n5-Fold CV NDCG@10 (Full Model): {full_res['cv_ndcg10_mean']:.4f} | 95% Job-Cluster CI: [{ci_cv[0]:.4f}, {ci_cv[1]:.4f}] (n=490 jobs)")

    # 7. ITEM 4: Provenance Table & Multi-System Accounting
    print("\n=== ITEM 4: Provenance Accounting ===", flush=True)
    prov_marginal = defaultdict(list)
    prov_exclusive = defaultdict(list)
    
    for _, r in dev_df.iterrows():
        lbl = r["human_label"]
        plist = list(r["provenance"])
        for sys_tag in plist:
            prov_marginal[sys_tag].append(lbl)
        if len(plist) == 1:
            prov_exclusive[plist[0]].append(lbl)
        else:
            prov_exclusive["Multiple"].append(lbl)
            
    prov_table = []
    for sys_tag in ["BM25", "BGE", "E5"]:
        arr = np.array(prov_marginal[sys_tag])
        n_ret = len(arr)
        mean_g = float(np.mean(arr))
        pct_ge2 = float(np.mean(arr >= 2) * 100)
        pct_eq3 = float(np.mean(arr == 3) * 100)
        prov_table.append({
            "system": sys_tag,
            "n_retrieved": n_ret,
            "mean_grade": mean_g,
            "pct_ge_2": pct_ge2,
            "pct_eq_3": pct_eq3
        })
    df_prov = pd.DataFrame(prov_table)
    print("--- Marginal Retrieval per System (includes candidates retrieved by >1 system) ---")
    print(df_prov.to_string(index=False))
    
    print("\n--- Mutually Exclusive Partition of 86 Dev Pairs ---")
    for k, v in prov_exclusive.items():
        arr = np.array(v)
        print(f"{k:<10}: n={len(arr):>2} | Mean={np.mean(arr):.2f} | % >= 2: {np.mean(arr>=2)*100:.1f}% | % = 3: {np.mean(arr==3)*100:.1f}%")

    # 8. ITEM 5: Judge vs Human Label Distribution & Quantile Sensitivity
    print("\n=== ITEM 5: Distribution Comparison & Quantile Sensitivity ===", flush=True)
    tr_dist = train_df["judge_label"].value_counts(normalize=True).sort_index().to_dict()
    dev_dist = dev_df["human_label"].value_counts(normalize=True).sort_index().to_dict()
    
    print("Judge Train Distribution (Rounded 0-3):")
    for g in range(4):
        print(f"  Grade {g}: {tr_dist.get(g, 0.0)*100:.1f}% ({train_df['judge_label'].value_counts()[g]} pairs)")
    print("Dev Human Distribution (0-3):")
    for g in range(4):
        print(f"  Grade {g}: {dev_dist.get(g, 0.0)*100:.1f}% ({dev_df['human_label'].value_counts()[g]} pairs)")
        
    print("\nRunning Sensitivity Variant: Quantile-Binned Expected Scores (4 bins)...", flush=True)
    sens_res = run_lambdarank(train_df, dev_df, exact_features, clean_folds, target_col="judge_quantile_label")
    print(f"Sensitivity CV NDCG@10:  {sens_res['cv_ndcg10_mean']:.4f} (vs Random {sens_res['cv_random_mean']:.4f})")
    print(f"Sensitivity Tuning:      {sens_res['tuning_ndcg10']:.4f} (vs Primary {full_res['tuning_ndcg10']:.4f})")
    print(f"Sensitivity Held-out:    {sens_res['heldout_ndcg10']:.4f} (vs Primary {full_res['heldout_ndcg10']:.4f})")
    print(f"Sensitivity All Dev:     {sens_res['all_dev_ndcg10']:.4f} (vs Primary {full_res['all_dev_ndcg10']:.4f})")

    # Save all results
    all_step1_outputs = {
        "feature_statistics": feat_stats,
        "ablations": {
            k: {
                "cv_ndcg10_mean": v["cv_ndcg10_mean"],
                "cv_random_mean": v["cv_random_mean"],
                "tuning_ndcg10": v["tuning_ndcg10"],
                "tuning_rand": v["tuning_rand"],
                "heldout_ndcg10": v["heldout_ndcg10"],
                "heldout_rand": v["heldout_rand"],
                "all_dev_ndcg10": v["all_dev_ndcg10"],
                "all_dev_rand": v["all_dev_rand"],
                "feature_importances": v["feature_importances"]
            } for k, v in ablation_res.items()
        },
        "cv_length_spearman": {
            "train_judge": {"spearman": float(tr_sp), "p": float(tr_sp_p)},
            "dev_human": {"spearman": float(dv_sp), "p": float(dv_sp_p)}
        },
        "paired_bootstrap_dev": {
            "mean_lambdamart": float(df_comp["lambdamart"].mean()),
            "mean_bge": float(df_comp["bge"].mean()),
            "mean_random": float(df_comp["random"].mean()),
            "diff_lm_bge": float(np.mean(diff_lm_bge)),
            "ci_lm_bge": [float(ci_lm_bge[0]), float(ci_lm_bge[1])],
            "diff_lm_rand": float(np.mean(diff_lm_rand)),
            "ci_lm_rand": [float(ci_lm_rand[0]), float(ci_lm_rand[1])],
            "diff_bge_rand": float(np.mean(diff_bge_rand)),
            "ci_bge_rand": [float(ci_bge_rand[0]), float(ci_bge_rand[1])],
            "per_job": comparison_table
        },
        "provenance_table": prov_table,
        "sensitivity_variant": {
            "cv_ndcg10_mean": sens_res["cv_ndcg10_mean"],
            "tuning_ndcg10": sens_res["tuning_ndcg10"],
            "heldout_ndcg10": sens_res["heldout_ndcg10"],
            "all_dev_ndcg10": sens_res["all_dev_ndcg10"]
        }
    }
    
    with open("data/processed/step1_revisions_summary.json", "w") as f:
        json.dump(all_step1_outputs, f, indent=2)
    print("\nSaved full outputs to data/processed/step1_revisions_summary.json", flush=True)

if __name__ == "__main__":
    main()
