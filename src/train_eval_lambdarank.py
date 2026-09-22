import os
import json
import warnings
import numpy as np
import pandas as pd
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
    mean_gain = np.mean(2**r - 1)
    k_eff = min(k, m)
    discount_sum = np.sum(1.0 / np.log2(np.arange(2, k_eff + 2)))
    return (mean_gain * discount_sum) / idcg

def evaluate_predictions(df_eval, score_col, label_col='human_label', k=10):
    """
    Computes per-job and mean NDCG@10, MRR (>=2, >=1) for predictions in df_eval.
    """
    job_ndcgs = []
    job_rands = []
    job_mrrs2 = []
    job_mrrs1 = []
    
    for jid, grp in df_eval.groupby("job_id"):
        grp_sorted = grp.sort_values(by=score_col, ascending=False)
        labels = grp_sorted[label_col].values
        
        ndcg = ndcg_at_k(labels, k=k)
        rand = expected_random_ndcg_at_k(labels, k=k)
        mrr2 = mrr_at_k(labels, k=k, rel_thresh=2)
        mrr1 = mrr_at_k(labels, k=k, rel_thresh=1)
        
        job_ndcgs.append(ndcg)
        job_rands.append(rand)
        job_mrrs2.append(mrr2)
        job_mrrs1.append(mrr1)
        
    return {
        "mean_ndcg10": float(np.mean(job_ndcgs)),
        "mean_random_ndcg10": float(np.mean(job_rands)),
        "mean_mrr_ge2": float(np.mean(job_mrrs2)),
        "mean_mrr_ge1": float(np.mean(job_mrrs1)),
        "per_job_ndcg10": [float(x) for x in job_ndcgs]
    }

def run_lambdarank_pipeline(train_df, dev_df, feature_cols, folds_dict):
    """
    Runs 5-fold CV on train_df using company-grouped folds,
    then trains full model on ALL train pairs and evaluates on dev_df (tuning, heldout, all).
    """
    # 1. 5-Fold Cross Validation
    fold_ndcgs = []
    fold_rands = []
    
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
    
    for fold_name, test_jids in folds_dict.items():
        test_mask = train_df["job_id"].isin(test_jids)
        tr_mask = ~test_mask
        
        df_tr = train_df[tr_mask].sort_values("job_id")
        df_te = train_df[test_mask].sort_values("job_id")
        
        if len(df_te) == 0:
            continue
            
        X_tr = df_tr[feature_cols]
        y_tr = df_tr["judge_label"].values
        groups_tr = df_tr.groupby("job_id", sort=False).size().values
        
        X_te = df_te[feature_cols]
        y_te = df_te["judge_label"].values
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
        
        eval_res = evaluate_predictions(df_te_eval, "pred", label_col="judge_label")
        fold_ndcgs.append(eval_res["mean_ndcg10"])
        fold_rands.append(eval_res["mean_random_ndcg10"])
        
    cv_mean_ndcg = float(np.mean(fold_ndcgs))
    cv_mean_rand = float(np.mean(fold_rands))
    
    # 2. Train on ALL train data
    df_all_tr = train_df.sort_values("job_id")
    X_all_tr = df_all_tr[feature_cols]
    y_all_tr = df_all_tr["judge_label"].values
    groups_all_tr = df_all_tr.groupby("job_id", sort=False).size().values
    
    full_model = lgb.LGBMRanker(**params)
    full_model.fit(X_all_tr, y_all_tr, group=groups_all_tr)
    
    # 3. Evaluate on Dev (tuning 45, heldout 41, all 86)
    dev_df_eval = dev_df.copy()
    dev_df_eval["pred"] = full_model.predict(dev_df[feature_cols])
    
    df_tuning = dev_df_eval[dev_df_eval["split"] == "tuning"]
    df_heldout = dev_df_eval[dev_df_eval["split"] == "heldout"]
    
    tuning_res = evaluate_predictions(df_tuning, "pred", label_col="human_label")
    heldout_res = evaluate_predictions(df_heldout, "pred", label_col="human_label")
    all_res = evaluate_predictions(dev_df_eval, "pred", label_col="human_label")
    
    # Feature importances
    importances = dict(zip(feature_cols, [float(x) for x in full_model.feature_importances_]))
    
    return {
        "cv_ndcg10_folds": fold_ndcgs,
        "cv_ndcg10_mean": cv_mean_ndcg,
        "cv_random_mean": cv_mean_rand,
        "dev_tuning": tuning_res,
        "dev_heldout": heldout_res,
        "dev_all": all_res,
        "feature_importances": importances
    }

def main():
    print("=== Training & Evaluating LightGBM LambdaMART ===", flush=True)
    judge_results_file = "data/processed/train_judge_results.jsonl"
    if not os.path.exists(judge_results_file):
        print(f"Error: {judge_results_file} not found. Please wait for judge run to finish.", flush=True)
        return
        
    judge_map = {}
    with open(judge_results_file) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("is_valid", True):
                pair_id = item["pair_id"]
                judge_map[pair_id] = {
                    "judge_label": item["avg_round"],
                    "judge_expected_score": item["avg_expected_score"]
                }
                
    print(f"Loaded {len(judge_map)} valid judge results.", flush=True)
    
    train_feat_path = "data/processed/features_train.parquet"
    train_df = pd.read_parquet(train_feat_path)
    train_df["pair_id"] = train_df["job_id"] + "_" + train_df["cv_id"]
    
    train_df = train_df[train_df["pair_id"].isin(judge_map)].copy()
    train_df["judge_label"] = train_df["pair_id"].map(lambda pid: judge_map[pid]["judge_label"])
    train_df["judge_expected_score"] = train_df["pair_id"].map(lambda pid: judge_map[pid]["judge_expected_score"])
    print(f"Train set with features and judge labels: {len(train_df)} pairs across {train_df['job_id'].nunique()} jobs.", flush=True)
    
    lbl_counts = train_df["judge_label"].value_counts().sort_index().to_dict()
    print("Train Judge Label Distribution (rounded 0-3):", lbl_counts, flush=True)
    
    dev_feat_path = "data/processed/features_dev_86.parquet"
    dev_df = pd.read_parquet(dev_feat_path)
    print(f"Dev set with features: {len(dev_df)} pairs ({sum(dev_df['split']=='tuning')} tuning, {sum(dev_df['split']=='heldout')} heldout).", flush=True)
    
    with open("data/processed/train_500_cv_folds.json") as f:
        folds = json.load(f)
    with open("data/processed/excluded_train_job_ids_v2.txt") as f:
        excluded = set(x.strip() for x in f if x.strip())
    clean_folds = {k: [j for j in v if j not in excluded] for k, v in folds.items()}
    
    groups = {
        "Lexical": [
            "bm25", "bm25_rank", "bm25_diff_top"
        ],
        "Dense": [
            "bge", "bge_rank", "bge_diff_top",
            "e5", "e5_rank", "e5_diff_top"
        ],
        "Structured": [
            "cv_token_len", "cv_token_len_rank", "cv_token_len_diff_top",
            "tech_overlap", "tech_overlap_rank", "tech_overlap_diff_top"
        ],
        "Lexical+Dense": [
            "bm25", "bm25_rank", "bm25_diff_top",
            "bge", "bge_rank", "bge_diff_top",
            "e5", "e5_rank", "e5_diff_top"
        ],
        "Full_Model (Lex+Dense+Struct)": [
            "bm25", "bm25_rank", "bm25_diff_top",
            "bge", "bge_rank", "bge_diff_top",
            "e5", "e5_rank", "e5_diff_top",
            "cv_token_len", "cv_token_len_rank", "cv_token_len_diff_top",
            "tech_overlap", "tech_overlap_rank", "tech_overlap_diff_top"
        ]
    }
    
    ablation_results = {}
    print("\n=== Feature-Group Ablation ===", flush=True)
    print(f"{'Feature Group':<30} | {'CV NDCG@10':<11} | {'Tuning (45)':<11} | {'Heldout (41)':<12} | {'All Dev (86)':<12}", flush=True)
    print("-" * 85, flush=True)
    
    for grp_name, feat_cols in groups.items():
        res = run_lambdarank_pipeline(train_df, dev_df, feat_cols, clean_folds)
        ablation_results[grp_name] = res
        print(f"{grp_name:<30} | {res['cv_ndcg10_mean']:.4f}      | {res['dev_tuning']['mean_ndcg10']:.4f}      | {res['dev_heldout']['mean_ndcg10']:.4f}       | {res['dev_all']['mean_ndcg10']:.4f}", flush=True)

    full_res = ablation_results["Full_Model (Lex+Dense+Struct)"]
    print(f"{'Random Baseline':<30} | {full_res['cv_random_mean']:.4f}      | {full_res['dev_tuning']['mean_random_ndcg10']:.4f}      | {full_res['dev_heldout']['mean_random_ndcg10']:.4f}       | {full_res['dev_all']['mean_random_ndcg10']:.4f}", flush=True)
    
    output_path = "data/processed/lambdarank_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nSaved LambdaMART ablation results to {output_path}", flush=True)

if __name__ == "__main__":
    main()
