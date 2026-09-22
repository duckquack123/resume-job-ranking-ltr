import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_TOKENS = {'0': 15, '1': 16, '2': 17, '3': 18}

def compute_expected_score(logits):
    probs = torch.softmax(logits, dim=-1)
    score_probs = {}
    prob_mass = 0.0
    for digit, tok_id in TARGET_TOKENS.items():
        p = probs[tok_id].item()
        score_probs[int(digit)] = p
        prob_mass += p
        
    if prob_mass > 0.01:
        renormalized = {k: v / prob_mass for k, v in score_probs.items()}
        expected = sum(k * v for k, v in renormalized.items())
        rounded = round(expected)
        return expected, rounded, renormalized, prob_mass, True
    else:
        return 0.0, -1, {}, prob_mass, False

def main():
    token = os.environ.get("HF_TOKEN")
    out_dir = "data/processed"
    pairs_file = f"{out_dir}/train_490_pairs_clean.parquet"
    results_file = f"{out_dir}/train_judge_results.jsonl"
    summary_file = f"{out_dir}/train_judge_summary.json"
    
    print(f"Loading pairs from {pairs_file}...")
    df_pairs = pd.read_parquet(pairs_file)
    print(f"Total training pairs to judge: {len(df_pairs)}")
    
    # Check already completed pairs for resumability
    completed_pids = set()
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_pids.add(record["pair_id"])
                    except Exception:
                        pass
        print(f"Resuming: found {len(completed_pids)} already judged pairs.")
    else:
        print("Starting fresh run.")
        
    remaining_pairs = df_pairs[~df_pairs.apply(lambda r: f"{r['job_id']}_{r['cv_id']}", axis=1).isin(completed_pids)].copy()
    print(f"Remaining pairs to evaluate: {len(remaining_pairs)}")
    
    if len(remaining_pairs) == 0:
        print("All pairs already evaluated. Computing summary...")
    else:
        # Load prompt template
        with open(f"{out_dir}/frozen_judge_prompt_template.txt") as f:
            prompt_v4_template = f.read().strip()
            
        # Load filtered text
        target_jids = set(remaining_pairs["job_id"].unique())
        target_cids = set(remaining_pairs["cv_id"].unique())
        
        print("Loading text for jobs and CVs...")
        jobs_table = pq.read_table(f"{out_dir}/jobs_train.parquet", columns=["id", "Position", "Long Description"])
        jobs_df = jobs_table.to_pandas()
        jobs_df = jobs_df[jobs_df["id"].isin(target_jids)]
        jobs_map = {r["id"]: {"title": str(r["Position"]), "desc": str(r["Long Description"])} for _, r in jobs_df.iterrows()}
        
        cvs_table = pq.read_table(f"{out_dir}/cvs_train.parquet", columns=["id", "CV"])
        cvs_df = cvs_table.to_pandas()
        cvs_df = cvs_df[cvs_df["id"].isin(target_cids)]
        cvs_map = {r["id"]: str(r["CV"]) if pd.notna(r["CV"]) else "" for _, r in cvs_df.iterrows()}
        
        print(f"Loaded {len(jobs_map)} unique jobs and {len(cvs_map)} unique CVs.")
        
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
        
        print(f"Running inference on {len(remaining_pairs)} pairs (both orders averaged)...")
        start_time = time.time()
        
        with open(results_file, "a") as f_out:
            for idx, (_, row) in enumerate(remaining_pairs.iterrows()):
                jid = row["job_id"]
                cid = row["cv_id"]
                pid = f"{jid}_{cid}"
                
                j_info = jobs_map.get(jid, {"title": "", "desc": ""})
                c_text = cvs_map.get(cid, "")
                
                # Order 1: Job then CV
                p_orig_body = f"""{prompt_v4_template}

Job Title: {j_info['title']}
Job Description:
{j_info['desc']}

CV:
{c_text}

Output ONLY a single integer (0, 1, 2, or 3) representing the score."""

                p_orig = tokenizer.apply_chat_template([{"role": "user", "content": p_orig_body}], tokenize=False, add_generation_prompt=True) + "Grade (0-3): "
                inp_orig = torch.tensor([tokenizer.encode(p_orig, add_special_tokens=False)]).to(model.device)
                
                # Order 2: CV then Job
                p_swap_body = f"""{prompt_v4_template}

CV:
{c_text}

Job Title: {j_info['title']}
Job Description:
{j_info['desc']}

Output ONLY a single integer (0, 1, 2, or 3) representing the score."""

                p_swap = tokenizer.apply_chat_template([{"role": "user", "content": p_swap_body}], tokenize=False, add_generation_prompt=True) + "Grade (0-3): "
                inp_swap = torch.tensor([tokenizer.encode(p_swap, add_special_tokens=False)]).to(model.device)
                
                with torch.inference_mode():
                    out_o = model(input_ids=inp_orig)
                    e_o, r_o, probs_o, pm_o, val_o = compute_expected_score(out_o.logits[0, -1, :])
                    
                    out_s = model(input_ids=inp_swap)
                    e_s, r_s, probs_s, pm_s, val_s = compute_expected_score(out_s.logits[0, -1, :])
                    
                is_valid = val_o and val_s
                if is_valid:
                    avg_exp = (e_o + e_s) / 2.0
                    avg_round = round(avg_exp)
                else:
                    avg_exp = -1.0
                    avg_round = -1
                    
                record = {
                    "pair_id": pid,
                    "job_id": jid,
                    "cv_id": cid,
                    "orig_prob_mass": pm_o,
                    "orig_expected_score": e_o,
                    "orig_round": r_o,
                    "orig_probs": probs_o,
                    "swap_prob_mass": pm_s,
                    "swap_expected_score": e_s,
                    "swap_round": r_s,
                    "swap_probs": probs_s,
                    "avg_expected_score": avg_exp,
                    "avg_round": avg_round,
                    "is_valid": is_valid
                }
                
                f_out.write(json.dumps(record) + "\n")
                
                if (idx + 1) % 50 == 0 or (idx + 1) == len(remaining_pairs):
                    f_out.flush()
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    rem_sec = (len(remaining_pairs) - (idx + 1)) / max(rate, 0.01)
                    print(f"[{idx+1}/{len(remaining_pairs)}] - Rate: {rate:.2f} pairs/s - ETA: {rem_sec/60:.1f} min")

    # Compute Summary
    print("\nComputing final training judge summary...")
    all_records = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))
                
    df_res = pd.DataFrame(all_records)
    total_pairs = len(df_res)
    valid_pairs = df_res[df_res["is_valid"] == True]
    n_valid = len(valid_pairs)
    n_invalid = total_pairs - n_valid
    
    grades = valid_pairs["avg_round"].value_counts().sort_index().to_dict()
    grade_dist = {int(k): int(v) for k, v in grades.items()}
    grade_pcts = {int(k): float(v / n_valid) for k, v in grades.items()}
    
    prob_masses = np.concatenate([df_res["orig_prob_mass"].values, df_res["swap_prob_mass"].values])
    pm_mean = float(np.mean(prob_masses))
    pm_p5 = float(np.percentile(prob_masses, 5))
    pm_min = float(np.min(prob_masses))
    
    summary = {
        "pairs_file": pairs_file,
        "total_pairs": total_pairs,
        "valid_pairs": n_valid,
        "invalid_pairs": n_invalid,
        "grade_distribution": grade_dist,
        "grade_percentages": grade_pcts,
        "digit_probability_mass": {
            "mean": pm_mean,
            "p5": pm_p5,
            "min": pm_min
        }
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("Training Judge Evaluation Complete!")
    print(f"Total Pairs: {total_pairs} (Valid: {n_valid}, Invalid: {n_invalid})")
    print(f"Grade Distribution (0-3): {grade_dist}")
    print(f"Grade Percentages: {grade_pcts}")
    print(f"Digit Prob Mass: Mean={pm_mean:.4f}, p5={pm_p5:.4f}, Min={pm_min:.4f}")

if __name__ == "__main__":
    main()
