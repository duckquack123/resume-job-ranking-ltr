import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score, cohen_kappa_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_TOKENS = {'0': 15, '1': 16, '2': 17, '3': 18}
EXPECTED_SHA_PREFIX = "5f852d90"
EXPECTED_SHA_SUFFIX = "7201d"

def compute_expected_score(logits):
    probs = torch.softmax(logits, dim=-1)
    score_probs = {}
    prob_mass = 0.0
    for digit, tok_id in TARGET_TOKENS.items():
        p = probs[tok_id].item()
        score_probs[int(digit)] = p
        prob_mass += p
        
    if prob_mass > 0:
        renormalized = {k: v / prob_mass for k, v in score_probs.items()}
        expected = sum(k * v for k, v in renormalized.items())
        rounded = round(expected)
        return expected, rounded, renormalized, prob_mass
    else:
        return 0.0, 0, {}, 0.0

def main():
    labels_file = "data/processed/labels_heldout.json"
    if not os.path.exists(labels_file):
        print(f"ERROR: {labels_file} does not exist.")
        sys.exit(1)
        
    with open(labels_file, "rb") as f:
        file_bytes = f.read()
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()
        
    print(f"Found {labels_file} with SHA-256: {sha256_hash}")
    if not (sha256_hash.startswith(EXPECTED_SHA_PREFIX) and sha256_hash.endswith(EXPECTED_SHA_SUFFIX)):
        print(f"ERROR: Hash {sha256_hash} does not match expected {EXPECTED_SHA_PREFIX}...{EXPECTED_SHA_SUFFIX}!")
        sys.exit(1)
    else:
        print(f"Verified SHA-256 matches expected ({EXPECTED_SHA_PREFIX}...{EXPECTED_SHA_SUFFIX}).")

    with open(labels_file) as f:
        labels_data = json.load(f)
        
    with open("data/processed/dev_hand_labeling_tasks.json") as f:
        tasks_data = json.load(f)
        
    task_map = {t["pair_id"]: t for t in tasks_data}
    
    with open("data/processed/frozen_judge_prompt_template.txt") as f:
        prompt_v4_template = f.read().strip()
        
    jobs_df = pd.read_parquet("data/processed/jobs_dev.parquet")
    rf_map = dict(zip(jobs_df["id"], jobs_df["Role_Family"]))
    
    token = os.environ.get("HF_TOKEN")
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token
    )
    model.eval()
    
    pair_meta = []
    print(f"Evaluating {len(labels_data)} held-out pairs...")
    
    for item in labels_data:
        pid = item["pair_id"]
        hand_s = item["score"]
        t = task_map[pid]
        jid = pid.split("_job_")[0] + "_job" if "_job_" in pid else pid.split("_")[0]
        role_fam = rf_map.get(jid, "unknown")
        
        # Original order
        prompt_orig_body = f"""{prompt_v4_template}

Job Title: {t['job_title']}
Job Description:
{t['job_desc']}

CV:
{t['cv_text']}

Output ONLY a single integer (0, 1, 2, or 3) representing the score."""

        p_orig = tokenizer.apply_chat_template([{"role": "user", "content": prompt_orig_body}], tokenize=False, add_generation_prompt=True) + "Grade (0-3): "
        inp_orig = torch.tensor([tokenizer.encode(p_orig, add_special_tokens=False)]).to(model.device)
        
        # Swapped order
        prompt_swap_body = f"""{prompt_v4_template}

CV:
{t['cv_text']}

Job Title: {t['job_title']}
Job Description:
{t['job_desc']}

Output ONLY a single integer (0, 1, 2, or 3) representing the score."""

        p_swap = tokenizer.apply_chat_template([{"role": "user", "content": prompt_swap_body}], tokenize=False, add_generation_prompt=True) + "Grade (0-3): "
        inp_swap = torch.tensor([tokenizer.encode(p_swap, add_special_tokens=False)]).to(model.device)
        
        with torch.inference_mode():
            out_o = model(input_ids=inp_orig)
            e_o, r_o, _, _ = compute_expected_score(out_o.logits[0, -1, :])
            
            out_s = model(input_ids=inp_swap)
            e_s, r_s, _, _ = compute_expected_score(out_s.logits[0, -1, :])
            
        avg_exp = (e_o + e_s) / 2.0
        avg_round = round(avg_exp)
        
        pair_meta.append({
            "pair_id": pid,
            "job_id": jid,
            "role_family": role_fam,
            "hand_score": hand_s,
            "orig_exp": e_o,
            "orig_round": r_o,
            "swap_exp": e_s,
            "swap_round": r_s,
            "avg_exp": avg_exp,
            "avg_round": avg_round
        })
        
    df = pd.DataFrame(pair_meta)
    
    hand = df["hand_score"].values
    avg_exp = df["avg_exp"].values
    avg_round = df["avg_round"].values
    orig_round = df["orig_round"].values
    orig_exp = df["orig_exp"].values
    
    # Order-averaged overall metrics
    qwk = float(cohen_kappa_score(hand, avg_round, weights="quadratic"))
    spearman_corr, _ = spearmanr(hand, avg_exp)
    exact_acc = float(np.mean(hand == avg_round))
    within_one = float(np.mean(np.abs(hand - avg_round) <= 1))
    conf_mat = confusion_matrix(hand, avg_round, labels=[0, 1, 2, 3]).tolist()
    
    # Single-order overall metrics
    qwk_single = float(cohen_kappa_score(hand, orig_round, weights="quadratic"))
    spearman_single, _ = spearmanr(hand, orig_exp)
    exact_acc_single = float(np.mean(hand == orig_round))
    within_one_single = float(np.mean(np.abs(hand - orig_round) <= 1))
    
    # Per-job metrics and random NDCG@10 baseline
    np.random.seed(42)
    per_job = []
    for jid, grp in df.groupby("job_id"):
        jh = grp["hand_score"].values
        je_avg = grp["avg_exp"].values
        je_orig = grp["orig_exp"].values
        n_j = len(grp)
        
        # Spearman
        if len(np.unique(jh)) > 1 and len(np.unique(je_avg)) > 1:
            sp_j_avg, _ = spearmanr(jh, je_avg)
        else:
            sp_j_avg = 0.0
            
        if len(np.unique(jh)) > 1 and len(np.unique(je_orig)) > 1:
            sp_j_orig, _ = spearmanr(jh, je_orig)
        else:
            sp_j_orig = 0.0
            
        # NDCG@10
        try:
            ndcg_avg = float(ndcg_score([jh], [je_avg], k=10))
            ndcg_orig = float(ndcg_score([jh], [je_orig], k=10))
        except Exception:
            ndcg_avg = 0.0
            ndcg_orig = 0.0
            
        # Expected random NDCG@10 baseline
        sim_ndcgs = []
        for _ in range(10000):
            rand_perm = np.random.permutation(n_j)
            sim_ndcgs.append(ndcg_score([jh], [rand_perm], k=10))
        rand_ndcg_exp = float(np.mean(sim_ndcgs))
        
        per_job.append({
            "job_id": jid,
            "role_family": grp["role_family"].iloc[0],
            "n": n_j,
            "spearman_avg": float(sp_j_avg),
            "spearman_orig": float(sp_j_orig),
            "ndcg10_avg": ndcg_avg,
            "ndcg10_orig": ndcg_orig,
            "random_ndcg10_expected": rand_ndcg_exp
        })
        
    mean_sp_avg = float(np.mean([p["spearman_avg"] for p in per_job]))
    mean_ndcg_avg = float(np.mean([p["ndcg10_avg"] for p in per_job]))
    mean_rand_ndcg = float(np.mean([p["random_ndcg10_expected"] for p in per_job]))
    
    # Per-role-family agreement (with n)
    per_rf = []
    for rf, grp in df.groupby("role_family"):
        jh = grp["hand_score"].values
        jr = grp["avg_round"].values
        exact_rf = float(np.mean(jh == jr))
        within_one_rf = float(np.mean(np.abs(jh - jr) <= 1))
        per_rf.append({
            "role_family": rf,
            "n": len(grp),
            "exact_agreement": exact_rf,
            "within_one_grade": within_one_rf,
            "hand_mean": float(np.mean(jh)),
            "judge_mean": float(np.mean(jr))
        })
        
    results = {
        "labels_file": labels_file,
        "labels_sha256": sha256_hash,
        "n_pairs": len(df),
        "order_averaged": {
            "qwk": qwk,
            "spearman": float(spearman_corr),
            "exact_agreement": exact_acc,
            "within_one_grade_rate": within_one,
            "mean_per_job_spearman": mean_sp_avg,
            "mean_per_job_ndcg10": mean_ndcg_avg,
            "mean_random_ndcg10_expected": mean_rand_ndcg
        },
        "single_order": {
            "qwk": qwk_single,
            "spearman": float(spearman_single),
            "exact_agreement": exact_acc_single,
            "within_one_grade_rate": within_one_single
        },
        "confusion_matrix": conf_mat,
        "per_job_metrics": per_job,
        "per_role_family_agreement": per_rf,
        "pairs": pair_meta
    }
    
    out_file = "data/processed/heldout_diagnostics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nHeldout Evaluation Complete. Results written to {out_file}")
    print(f"QWK (order-avg): {qwk:.4f} (single: {qwk_single:.4f})")
    print(f"Spearman (order-avg): {spearman_corr:.4f} (single: {spearman_single:.4f})")
    print(f"Exact Agreement: {exact_acc:.4f} ({np.sum(hand == avg_round)}/{len(df)})")
    print(f"Within-one-grade: {within_one:.4f} ({np.sum(np.abs(hand - avg_round) <= 1)}/{len(df)})")
    print(f"Mean Per-Job Spearman: {mean_sp_avg:.4f}")
    print(f"Mean Per-Job NDCG@10: {mean_ndcg_avg:.4f} vs Random NDCG@10: {mean_rand_ndcg:.4f}")
    print("\nConfusion Matrix (Hand Rows x Judge Cols):")
    for r in conf_mat:
        print(" ", r)
    print("\nPer-Job Details:")
    for pj in per_job:
        print(f"  Job {pj['job_id']} ({pj['role_family']}, n={pj['n']}): Spearman={pj['spearman_avg']:.4f}, NDCG@10={pj['ndcg10_avg']:.4f}, RandNDCG={pj['random_ndcg10_expected']:.4f}")
    print("\nPer-Role-Family Details:")
    for pr in per_rf:
        print(f"  Role {pr['role_family']} (n={pr['n']}): Exact={pr['exact_agreement']*100:.1f}%, Within1={pr['within_one_grade']*100:.1f}%")

if __name__ == "__main__":
    main()
