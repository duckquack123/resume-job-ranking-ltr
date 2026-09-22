import os
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score, cohen_kappa_score
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_TOKENS = {'0': 15, '1': 16, '2': 17, '3': 18}

PROMPT_V3 = """You are an expert technical recruiter evaluating a candidate's CV against a Job Description.
Evaluate candidate suitability strictly on technical alignment using this rubric:
0: Irrelevant. Candidate lacks the essential programming languages, foundational technologies, and core domain required by the job description.
1: Marginally Relevant. Candidate demonstrates basic programming knowledge or tangential tools, but lacks key mandatory frameworks and direct production experience in the core stack.
2: Somewhat Relevant. Candidate demonstrates solid competency in the required core programming language and primary frameworks, satisfying the main job requirements.
3: Highly Relevant. Candidate demonstrates comprehensive proficiency across both core and preferred technical competencies, with strong, proven experience matching the role.

Rules: Strictly ignore candidate name, gender, age, graduation year, education prestige, geographic location, remote preferences, and salary expectations.

Job Title: {title}
Job Description:
{desc}

CV:
{cv}

Output ONLY a single integer (0, 1, 2, or 3) representing the score."""

PROMPT_V4 = """You are an expert technical recruiter evaluating a candidate's CV against a Job Description.
Score the candidate match from 0 to 3 based on core technical competencies and experience:
0: Irrelevant. Candidate has no meaningful alignment with the primary technical requirements or engineering domain.
1: Marginally Relevant. Candidate has minor skill overlap or junior experience, but lacks the core language, primary framework, or necessary seniority.
2: Somewhat Relevant. Candidate satisfies the core technical requirements and primary programming language, demonstrating capable competency to perform the role.
3: Highly Relevant. Candidate is a strong, complete match for the position, with deep relevant experience in the core stack and key responsibilities.

Rules: Evaluate purely on technical fit. You MUST ignore name, gender, age, graduation year, nationality, institution prestige, location, remote preferences, and compensation.

Job Title: {title}
Job Description:
{desc}

CV:
{cv}

Output ONLY a single integer (0, 1, 2, or 3) representing the score."""

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
    token = os.environ.get("HF_TOKEN")
    
    # 1. Load labels and tasks
    labels_file = "data/processed/labels_tuning.json"
    with open(labels_file, "rb") as f:
        labels_sha256 = hashlib.sha256(f.read()).hexdigest()
        
    with open(labels_file) as f:
        labels_data = json.load(f)
        
    with open("data/processed/dev_hand_labeling_tasks.json") as f:
        tasks_data = json.load(f)
        
    task_map = {t["pair_id"]: t for t in tasks_data}
    
    # 2. Run inference for both orders of v4
    print("Loading tokenizer and model...")
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
    
    orig_exp = []
    swap_exp = []
    pair_meta = []
    
    for item in labels_data:
        pid = item["pair_id"]
        t = task_map[pid]
        hand_s = item["score"]
        jid = pid.split("_job_")[0] + "_job" if "_job_" in pid else pid.split("_")[0]
        
        # Original order
        u_orig = PROMPT_V4.format(title=t["job_title"], desc=t["job_desc"], cv=t["cv_text"])
        p_orig = tokenizer.apply_chat_template([{"role": "user", "content": u_orig}], tokenize=False, add_generation_prompt=True) + "Grade (0-3): "
        inp_orig = torch.tensor([tokenizer.encode(p_orig, add_special_tokens=False)]).to(model.device)
        
        # Swapped order
        u_swap = PROMPT_V4.replace(
            "Job Title: {title}\nJob Description:\n{desc}\n\nCV:\n{cv}",
            "CV:\n{cv}\n\nJob Title: {title}\nJob Description:\n{desc}"
        ).format(title=t["job_title"], desc=t["job_desc"], cv=t["cv_text"])
        p_swap = tokenizer.apply_chat_template([{"role": "user", "content": u_swap}], tokenize=False, add_generation_prompt=True) + "Grade (0-3): "
        inp_swap = torch.tensor([tokenizer.encode(p_swap, add_special_tokens=False)]).to(model.device)
        
        with torch.inference_mode():
            out_o = model(input_ids=inp_orig)
            e_o, r_o, _, _ = compute_expected_score(out_o.logits[0, -1, :])
            
            out_s = model(input_ids=inp_swap)
            e_s, r_s, _, _ = compute_expected_score(out_s.logits[0, -1, :])
            
        orig_exp.append(e_o)
        swap_exp.append(e_s)
        pair_meta.append({
            "pair_id": pid,
            "job_id": jid,
            "hand_score": hand_s,
            "orig_exp": e_o,
            "orig_round": r_o,
            "swap_exp": e_s,
            "swap_round": r_s,
            "avg_exp": (e_o + e_s) / 2.0,
            "avg_round": round((e_o + e_s) / 2.0)
        })
        
    df = pd.DataFrame(pair_meta)
    
    # Order-averaged metrics on 45 tuning pairs
    hand = df["hand_score"].values
    avg_exp = df["avg_exp"].values
    avg_round = df["avg_round"].values
    orig_round = df["orig_round"].values
    
    qwk_avg = float(cohen_kappa_score(hand, avg_round, weights="quadratic"))
    spearman_avg, _ = spearmanr(hand, avg_exp)
    exact_acc_avg = float(np.mean(hand == avg_round))
    within_one_avg = float(np.mean(np.abs(hand - avg_round) <= 1))
    
    # Original single-order metrics
    qwk_orig = float(cohen_kappa_score(hand, orig_round, weights="quadratic"))
    spearman_orig, _ = spearmanr(hand, df["orig_exp"].values)
    exact_acc_orig = float(np.mean(hand == orig_round))
    within_one_orig = float(np.mean(np.abs(hand - orig_round) <= 1))
    
    # Pairs with 2+ grade difference
    diff_2plus = df[np.abs(df["hand_score"] - df["orig_round"]) >= 2]
    diff_2plus_avg = df[np.abs(df["hand_score"] - df["avg_round"]) >= 2]
    
    # Per-job Spearman and NDCG@10
    per_job_metrics = []
    for jid, grp in df.groupby("job_id"):
        j_hand = grp["hand_score"].values
        j_pred = grp["orig_exp"].values
        j_avg = grp["avg_exp"].values
        
        # Spearman
        if len(np.unique(j_hand)) > 1 and len(np.unique(j_pred)) > 1:
            sp_j, _ = spearmanr(j_hand, j_pred)
            sp_j_avg, _ = spearmanr(j_hand, j_avg)
        else:
            sp_j = 0.0
            sp_j_avg = 0.0
            
        # NDCG@10 (needs 2D array: true_scores, pred_scores)
        try:
            ndcg_j = ndcg_score([j_hand], [j_pred], k=10)
            ndcg_j_avg = ndcg_score([j_hand], [j_avg], k=10)
        except Exception:
            ndcg_j = 0.0
            ndcg_j_avg = 0.0
            
        per_job_metrics.append({
            "job_id": jid,
            "n_pairs": len(grp),
            "spearman_orig": float(sp_j),
            "spearman_avg": float(sp_j_avg),
            "ndcg10_orig": float(ndcg_j),
            "ndcg10_avg": float(ndcg_j_avg)
        })
        
    # Overall NDCG@10 (mean across jobs)
    mean_ndcg10_orig = float(np.mean([m["ndcg10_orig"] for m in per_job_metrics]))
    mean_ndcg10_avg = float(np.mean([m["ndcg10_avg"] for m in per_job_metrics]))
    mean_sp_orig = float(np.mean([m["spearman_orig"] for m in per_job_metrics]))
    mean_sp_avg = float(np.mean([m["spearman_avg"] for m in per_job_metrics]))
    
    # Bootstrap 95% CI across pairs (1,000 resamples)
    np.random.seed(42)
    boot_spearmans = []
    boot_ndcgs = []
    boot_qwks = []
    
    n_boot = 1000
    n = len(df)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        sample = df.iloc[idx]
        sh = sample["hand_score"].values
        sp = sample["orig_exp"].values
        sr = sample["orig_round"].values
        
        if len(np.unique(sh)) > 1 and len(np.unique(sp)) > 1:
            boot_spearmans.append(spearmanr(sh, sp)[0])
            boot_qwks.append(cohen_kappa_score(sh, sr, weights="quadratic"))
            
    ci_sp = [float(np.percentile(boot_spearmans, 2.5)), float(np.percentile(boot_spearmans, 97.5))]
    ci_qwk = [float(np.percentile(boot_qwks, 2.5)), float(np.percentile(boot_qwks, 97.5))]
    
    out_data = {
        "labels_file": labels_file,
        "labels_sha256": labels_sha256,
        "tuning_order_averaged": {
            "qwk": qwk_avg,
            "spearman": float(spearman_avg),
            "exact_agreement": exact_acc_avg,
            "within_one_grade": within_one_avg
        },
        "tuning_single_order": {
            "qwk": qwk_orig,
            "spearman": float(spearman_orig),
            "exact_agreement": exact_acc_orig,
            "within_one_grade": within_one_orig
        },
        "bootstrap_ci_95": {
            "spearman": ci_sp,
            "qwk": ci_qwk
        },
        "per_job_metrics": per_job_metrics,
        "mean_per_job": {
            "spearman_orig": mean_sp_orig,
            "spearman_avg": mean_sp_avg,
            "ndcg10_orig": mean_ndcg10_orig,
            "ndcg10_avg": mean_ndcg10_avg
        },
        "diff_2plus_pairs_orig": [
            {"pair_id": r["pair_id"], "hand_grade": int(r["hand_score"]), "judge_grade": int(r["orig_round"])}
            for _, r in diff_2plus.iterrows()
        ],
        "diff_2plus_pairs_avg": [
            {"pair_id": r["pair_id"], "hand_grade": int(r["hand_score"]), "judge_grade": int(r["avg_round"])}
            for _, r in diff_2plus_avg.iterrows()
        ]
    }
    
    with open("data/processed/tuning_diagnostics.json", "w") as f:
        json.dump(out_data, f, indent=2)
        
    print("Diagnostics computed successfully!")
    print(f"Labels SHA-256: {labels_sha256}")
    print(f"Single Order QWK: {qwk_orig:.4f}, Order-Averaged QWK: {qwk_avg:.4f}")
    print(f"Single Order Exact: {exact_acc_orig:.4f}, Order-Averaged Exact: {exact_acc_avg:.4f}")
    print(f"Pairs with 2+ grade diff (single order): {len(diff_2plus)}")
    for _, r in diff_2plus.iterrows():
        print(f"  {r['pair_id']}: Hand={r['hand_score']} vs Judge={r['orig_round']}")

if __name__ == "__main__":
    main()
