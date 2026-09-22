import os
import json
import hashlib
import pandas as pd
import numpy as np
import lightgbm as lgb

def main():
    print("=== Freezing Step 1 LambdaMART Model & Assets ===")
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Feature list (15 exact scale-free features)
    feature_list = [
        "bm25_zscore", "bm25_ratio_top", "bm25_rank",
        "bge_zscore", "bge_ratio_top", "bge_rank",
        "e5_zscore", "e5_ratio_top", "e5_rank",
        "cv_token_len_zscore", "cv_token_len_ratio_top", "cv_token_len_rank",
        "tech_overlap_zscore", "tech_overlap_ratio_top", "tech_overlap_rank"
    ]
    feat_file = "data/processed/frozen_feature_list.json"
    with open(feat_file, "w") as f:
        json.dump(feature_list, f, indent=2)
    print(f"Saved {feat_file}")
    
    # 2. LightGBM config
    lgb_config = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 10,
        "n_estimators": 100,
        "early_stopping_rounds": 15,
        "random_state": 42,
        "n_jobs": 4,
        "verbose": -1
    }
    cfg_file = "data/processed/frozen_lgbm_config.json"
    with open(cfg_file, "w") as f:
        json.dump(lgb_config, f, indent=2)
    print(f"Saved {cfg_file}")
    
    # 3. Load training data (4,283 clean pairs)
    train_df = pd.read_parquet("data/processed/features_train_scale_free.parquet")
    print(f"Loaded train_df: {len(train_df)} rows, {train_df['job_id'].nunique()} jobs")
    
    # Verify target column: avg_round (rounded average grade 0-3)
    target_col = "judge_label"
    print(f"Target distribution in train_df:\n{train_df[target_col].value_counts().sort_index()}")
    
    # Sort by job_id for LightGBM ranking group order
    df_sorted = train_df.sort_values("job_id").reset_index(drop=True)
    X = df_sorted[feature_list]
    y = df_sorted[target_col].values
    groups = df_sorted.groupby("job_id", sort=False).size().values
    
    # Train full model
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[10],
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        n_estimators=100,
        random_state=42,
        n_jobs=4,
        verbose=-1
    )
    model.fit(X, y, group=groups)
    
    # Save model
    model_file = "models/lambdamart_step1_frozen.txt"
    model.booster_.save_model(model_file)
    print(f"Saved trained booster to {model_file} (trees: {model.booster_.num_trees()})")
    
    # Verify predictions on dev
    dev_df = pd.read_parquet("data/processed/features_dev_86_scale_free.parquet")
    dev_preds = model.predict(dev_df[feature_list])
    print(f"Dev predictions generated for {len(dev_preds)} pairs (mean={np.mean(dev_preds):.4f})")
    
    # Check hashes of all frozen assets
    assets = [
        ("Feature List", feat_file),
        ("LightGBM Config", cfg_file),
        ("Fold File", "data/processed/train_500_cv_folds.json"),
        ("Excluded Job IDs v2", "data/processed/excluded_train_job_ids_v2.txt"),
        ("Judge Label File", "data/processed/train_judge_results.jsonl"),
        ("Train Features Parquet", "data/processed/features_train_scale_free.parquet"),
        ("Trained Model", model_file)
    ]
    
    print("\n--- Frozen Assets & Hashes ---")
    hashes = {}
    for name, path in assets:
        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        hashes[name] = {"path": path, "sha256": h}
        print(f"{name:<25}: {h}  ({path})")
        
    with open("data/processed/frozen_step1_hashes.json", "w") as f:
        json.dump(hashes, f, indent=2)

if __name__ == "__main__":
    main()
