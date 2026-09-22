import os
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_expected_score(logits, target_tokens):
    probs = torch.softmax(logits, dim=-1)
    score_probs = {}
    prob_mass = 0.0
    for digit, tok_id in target_tokens.items():
        p = probs[tok_id].item()
        score_probs[int(digit)] = p
        prob_mass += p
        
    if prob_mass > 0:
        renormalized = {k: v / prob_mass for k, v in score_probs.items()}
        expected = sum(k * v for k, v in renormalized.items())
        rounded = round(expected)
        return expected, rounded, renormalized, prob_mass
    else:
        return None, None, {}, 0.0

def build_swapped_prompt(title, desc, cv, tokenizer):
    user_content = f"""You are an expert HR recruiter evaluating a candidate's CV against a Job Description.
Score the CV from 0 to 3 based on the following rubric:
0: Irrelevant. The candidate lacks the core skills or experience.
1: Marginally Relevant. The candidate has some overlapping skills but falls short on key requirements.
2: Somewhat Relevant. The candidate meets most requirements and could be considered.
3: Highly Relevant. The candidate is a strong match for the role.

Rules: You MUST ignore name, gender, age, graduation year, nationality, institution prestige, location, remote preferences and salary.

CV:
{cv}

Job Title: {title}
Job Description:
{desc}

Output ONLY a single integer (0, 1, 2, or 3) representing the score."""
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Grade (0-3): "
    return prompt

def main():
    token = os.environ.get("HF_TOKEN")
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Load original results
    original_results = {}
    with open("data/processed/timing_results.jsonl") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                original_results[d["pair_id"]] = d

    # Load tasks
    with open("data/processed/timing_tasks.json") as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} tasks and {len(original_results)} original results.")
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token
    )
    model.eval()

    target_tokens = {'0': 15, '1': 16, '2': 17, '3': 18}

    # Run inference on swapped prompts (CV first)
    swapped_scores = []
    orig_scores = []
    cv_char_lengths = []
    cv_token_lengths = []
    pair_records = []
    
    print("Running inference on order-swapped prompts...")
    for t in tasks:
        pid = t["pair_id"]
        orig_score = original_results[pid]["expected_score"]
        
        title = t.get("job_title", "")
        desc = t.get("job_desc", "")
        cv = t.get("cv_text", "")
        
        swapped_prompt = build_swapped_prompt(title, desc, cv, tokenizer)
        input_ids = tokenizer.encode(swapped_prompt, add_special_tokens=False)
        input_ids_tensor = torch.tensor([input_ids]).to(model.device)
        
        with torch.inference_mode():
            outputs = model(input_ids=input_ids_tensor)
            logits = outputs.logits[0, -1, :]
            swapped_exp, rounded, probs, prob_mass = compute_expected_score(logits, target_tokens)
            
        cv_tokens = len(tokenizer.encode(cv, add_special_tokens=False))
        cv_chars = len(cv)
        
        swapped_scores.append(swapped_exp)
        orig_scores.append(orig_score)
        cv_char_lengths.append(cv_chars)
        cv_token_lengths.append(cv_tokens)
        
        pair_records.append({
            "pair_id": pid,
            "original_expected_score": orig_score,
            "swapped_expected_score": swapped_exp,
            "abs_diff": abs(orig_score - swapped_exp),
            "cv_chars": cv_chars,
            "cv_tokens": cv_tokens
        })

    orig_arr = np.array(orig_scores)
    swap_arr = np.array(swapped_scores)
    abs_diffs = np.abs(orig_arr - swap_arr)

    # 1. Order-swap consistency
    spearman_corr, spearman_p = spearmanr(orig_arr, swap_arr)
    pearson_corr, pearson_p = pearsonr(orig_arr, swap_arr)
    mad = np.mean(abs_diffs)
    
    print("\n=== ORDER-SWAP CONSISTENCY (N=100) ===")
    print(f"Spearman rank correlation: {spearman_corr:.4f} (p = {spearman_p:.2e})")
    print(f"Pearson correlation: {pearson_corr:.4f} (p = {pearson_p:.2e})")
    print(f"Mean Absolute Difference (MAD): {mad:.4f}")
    print(f"Max Absolute Difference: {np.max(abs_diffs):.4f}")
    print(f"Original Mean Score: {np.mean(orig_arr):.4f}, Swapped Mean Score: {np.mean(swap_arr):.4f}")

    # 2. Correlation with CV length
    corr_chars_spearman, p_chars_s = spearmanr(orig_arr, cv_char_lengths)
    corr_chars_pearson, p_chars_p = pearsonr(orig_arr, cv_char_lengths)
    corr_toks_spearman, p_toks_s = spearmanr(orig_arr, cv_token_lengths)
    corr_toks_pearson, p_toks_p = pearsonr(orig_arr, cv_token_lengths)

    print("\n=== CORRELATION WITH CV LENGTH ===")
    print(f"Expected Score vs CV Chars - Spearman: {corr_chars_spearman:.4f} (p = {p_chars_s:.4f}), Pearson: {corr_chars_pearson:.4f} (p = {p_chars_p:.4f})")
    print(f"Expected Score vs CV Tokens - Spearman: {corr_toks_spearman:.4f} (p = {p_toks_s:.4f}), Pearson: {corr_toks_pearson:.4f} (p = {p_toks_p:.4f})")

    # 3. Score distributions per role family
    jobs_df = pd.read_parquet("data/processed/jobs_train.parquet")
    # Map job id to role family
    job_rf_map = {}
    for _, row in jobs_df.iterrows():
        job_rf_map[str(row["id"])] = str(row["Role_Family"])

    role_family_scores = {}
    for t, orig_s in zip(tasks, orig_scores):
        pid = t["pair_id"]
        jid = pid.split("_job_")[0] + "_job" if "_job_" in pid else pid.split("_")[0]
        # In timing_tasks.json, pair_id format is e.g. "0_job_8227_cv" or "0_8227"
        # Let's extract job id properly:
        jid_candidate = pid.split("_")[0]
        rf = job_rf_map.get(jid_candidate, job_rf_map.get(f"{jid_candidate}_job", "unknown"))
        if rf not in role_family_scores:
            role_family_scores[rf] = []
        role_family_scores[rf].append(orig_s)

    print("\n=== SCORE DISTRIBUTIONS PER ROLE FAMILY ===")
    rf_summary = {}
    for rf, s_list in sorted(role_family_scores.items()):
        arr = np.array(s_list)
        rf_summary[rf] = {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }
        print(f"Role Family: {rf} (N={len(arr)}) -> Mean: {np.mean(arr):.2f}, Std: {np.std(arr):.2f}, Min: {np.min(arr):.2f}, Max: {np.max(arr):.2f}")

    # Signed diff, flip fraction, and transition matrix
    signed_diff = float(np.mean(orig_arr - swap_arr))
    orig_rounded = np.round(orig_arr).astype(int)
    swap_rounded = np.round(swap_arr).astype(int)
    flip_fraction = float(np.mean(orig_rounded != swap_rounded))

    transitions = {g: {g2: 0 for g2 in range(4)} for g in range(4)}
    for g_orig, g_swap in zip(orig_rounded, swap_rounded):
        transitions[int(g_orig)][int(g_swap)] += 1

    print("\n=== ORDER-SWAP DETAILED METRICS ===")
    print(f"Mean Signed Difference (Orig - Swapped): {signed_diff:.4f}")
    print(f"Fraction of Rounded Grade Flips: {flip_fraction:.4f} ({int(flip_fraction * len(orig_arr))}/{len(orig_arr)})")
    print("Per-Grade Transitions (rows = Orig, cols = Swapped):")
    for g_orig in range(4):
        row_str = " | ".join(f"{g_orig}->{g_swap}: {transitions[g_orig][g_swap]:2d}" for g_swap in range(4))
        print(f"  Orig {g_orig}: {row_str}")

    # Save summary
    out_summary = {
        "order_swap": {
            "spearman": float(spearman_corr),
            "spearman_p": float(spearman_p),
            "pearson": float(pearson_corr),
            "mad": float(mad),
            "mean_signed_diff": float(signed_diff),
            "flip_fraction": float(flip_fraction),
            "transitions": transitions
        },
        "length_correlation": {
            "spearman_tokens": float(corr_toks_spearman),
            "pearson_tokens": float(corr_toks_pearson),
            "spearman_chars": float(corr_chars_spearman),
            "pearson_chars": float(corr_chars_pearson)
        },
        "role_family_distributions": rf_summary
    }

    with open("data/processed/order_swap_results.json", "w") as f:
        json.dump(out_summary, f, indent=2)

    with open("data/processed/order_swap_pairs.json", "w") as f:
        json.dump(pair_records, f, indent=2)

    print("\nResults saved to data/processed/order_swap_results.json and order_swap_pairs.json.")

if __name__ == "__main__":
    main()
