import os
import json
import hashlib
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score, confusion_matrix
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
        
    if prob_mass > 0:
        renormalized = {k: v / prob_mass for k, v in score_probs.items()}
        expected = sum(k * v for k, v in renormalized.items())
        rounded = round(expected)
        return expected, rounded, renormalized, prob_mass
    else:
        return 0.0, 0, {}, 0.0

# Define 4 prompt templates
PROMPT_V1 = """You are an expert HR recruiter evaluating a candidate's CV against a Job Description.
Score the CV from 0 to 3 based on the following rubric:
0: Irrelevant. The candidate lacks the core skills or experience.
1: Marginally Relevant. The candidate has some overlapping skills but falls short on key requirements.
2: Somewhat Relevant. The candidate meets most requirements and could be considered.
3: Highly Relevant. The candidate is a strong match for the role.

Rules: You MUST ignore name, gender, age, graduation year, nationality, institution prestige, location, remote preferences and salary.

Job Title: {title}
Job Description:
{desc}

CV:
{cv}

Output ONLY a single integer (0, 1, 2, or 3) representing the score."""

PROMPT_V2 = """You are an expert technical recruiter evaluating a candidate's CV against a Job Description.
Score the CV from 0 to 3 based on the following rubric:
0: Irrelevant. The candidate lacks the required core technology stack, programming language, or primary engineering background.
1: Marginally Relevant. The candidate has adjacent technical skills or secondary tool overlap, but falls short on mandatory core technologies or required years of relevant hands-on experience.
2: Somewhat Relevant. The candidate meets the essential core technical requirements and primary programming stack, with minor gaps only in secondary tools or domain specifics.
3: Highly Relevant. The candidate is a strong technical match who directly fulfills the primary tech stack, frameworks, and practical experience needed for the position.

Rules: You MUST ignore name, gender, age, graduation year, nationality, institution prestige, location, remote preferences and salary.

Job Title: {title}
Job Description:
{desc}

CV:
{cv}

Output ONLY a single integer (0, 1, 2, or 3) representing the score."""

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

VERSIONS = {
    "v1_baseline": PROMPT_V1,
    "v2_tech_stack_clarity": PROMPT_V2,
    "v3_competency_alignment": PROMPT_V3,
    "v4_seniority_requirements": PROMPT_V4
}

def format_prompt(template, title, desc, cv, tokenizer, swap_order=False):
    if swap_order:
        # Put CV before Job Description
        user_content = template.replace(
            "Job Title: {title}\nJob Description:\n{desc}\n\nCV:\n{cv}",
            "CV:\n{cv}\n\nJob Title: {title}\nJob Description:\n{desc}"
        ).format(title=title, desc=desc, cv=cv)
    else:
        user_content = template.format(title=title, desc=desc, cv=cv)
        
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Grade (0-3): "
    return prompt

def main():
    token = os.environ.get("HF_TOKEN")
    
    # Load tasks and labels
    with open("data/processed/labels_tuning.json") as f:
        labels_data = json.load(f)
        
    with open("data/processed/dev_hand_labeling_tasks.json") as f:
        tasks_data = json.load(f)
        
    task_map = {t["pair_id"]: t for t in tasks_data}
    
    # Load jobs metadata for role families
    jobs_df = pd.read_parquet("data/processed/jobs_dev.parquet")
    job_rf_map = dict(zip(jobs_df["id"].astype(str), jobs_df["Role_Family"]))
    
    tuning_pairs = []
    hand_labels = []
    for l in labels_data:
        pid = l["pair_id"]
        t = task_map[pid]
        jid = pid.split("_job_")[0] + "_job" if "_job_" in pid else pid.split("_")[0]
        rf = job_rf_map.get(jid, "unknown")
        
        tuning_pairs.append({
            "pair_id": pid,
            "job_id": jid,
            "role_family": rf,
            "job_title": t["job_title"],
            "job_desc": t["job_desc"],
            "cv_text": t["cv_text"],
            "hand_label": l["score"]
        })
        hand_labels.append(l["score"])
        
    hand_labels = np.array(hand_labels)
    n_pairs = len(tuning_pairs)
    print(f"Loaded {n_pairs} tuning pairs.")
    print(f"Hand labels distribution: {np.bincount(hand_labels, minlength=4)}")

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

    # Measure CV lengths
    cv_token_lens = [len(tokenizer.encode(p["cv_text"], add_special_tokens=False)) for p in tuning_pairs]
    cv_char_lens = [len(p["cv_text"]) for p in tuning_pairs]
    
    # Calculate hand label correlation with CV length
    hand_cv_tok_spearman, hand_cv_tok_p = spearmanr(hand_labels, cv_token_lens)
    hand_cv_char_spearman, hand_cv_char_p = spearmanr(hand_labels, cv_char_lens)
    print(f"\nHand Labels vs CV Token Length: Spearman r = {hand_cv_tok_spearman:.4f} (p = {hand_cv_tok_p:.4f})")
    print(f"Hand Labels vs CV Char Length: Spearman r = {hand_cv_char_spearman:.4f} (p = {hand_cv_char_p:.4f})")

    results = {}
    best_v = None
    best_qwk = -1.0

    for v_name, template_str in VERSIONS.items():
        template_hash = hashlib.sha256(template_str.encode("utf-8")).hexdigest()
        print(f"\nEvaluating {v_name} (SHA-256: {template_hash})...")
        
        expected_scores = []
        rounded_scores = []
        prob_masses = []
        
        for p in tuning_pairs:
            prompt = format_prompt(template_str, p["job_title"], p["job_desc"], p["cv_text"], tokenizer)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids_tensor = torch.tensor([input_ids]).to(model.device)
            
            with torch.inference_mode():
                outputs = model(input_ids=input_ids_tensor)
                logits = outputs.logits[0, -1, :]
                exp_s, round_s, probs, p_mass = compute_expected_score(logits)
                
            expected_scores.append(exp_s)
            rounded_scores.append(round_s)
            prob_masses.append(p_mass)
            
        exp_arr = np.array(expected_scores)
        round_arr = np.array(rounded_scores)
        
        qwk = float(cohen_kappa_score(hand_labels, round_arr, weights="quadratic"))
        spearman_corr, spearman_p = spearmanr(hand_labels, exp_arr)
        exact_acc = float(np.mean(hand_labels == round_arr))
        conf_mat = confusion_matrix(hand_labels, round_arr, labels=[0, 1, 2, 3]).tolist()
        
        # CV length correlation for this version
        judge_cv_tok_spearman, _ = spearmanr(exp_arr, cv_token_lens)
        judge_cv_char_spearman, _ = spearmanr(exp_arr, cv_char_lens)
        
        print(f"  QWK: {qwk:.4f}")
        print(f"  Spearman: {spearman_corr:.4f}")
        print(f"  Exact Agreement: {exact_acc:.4f} ({int(exact_acc * n_pairs)}/{n_pairs})")
        print(f"  Mean Prob Mass: {np.mean(prob_masses):.4f}")
        print(f"  Judge vs CV Tokens Spearman: {judge_cv_tok_spearman:.4f}")
        
        results[v_name] = {
            "template_hash": template_hash,
            "qwk": qwk,
            "spearman": float(spearman_corr),
            "exact_agreement": exact_acc,
            "mean_prob_mass": float(np.mean(prob_masses)),
            "confusion_matrix": conf_mat,
            "expected_scores": expected_scores,
            "rounded_scores": rounded_scores,
            "judge_cv_tok_spearman": float(judge_cv_tok_spearman),
            "judge_cv_char_spearman": float(judge_cv_char_spearman)
        }
        
        if qwk > best_qwk:
            best_qwk = qwk
            best_v = v_name

    print(f"\nBest prompt version by QWK: {best_v} (QWK = {best_qwk:.4f})")

    # Now run Order-Swap bias check on Baseline (v1) and Best Version
    print("\n--- Running Bias & Order-Swap Checks on Tuning Pairs ---")
    bias_checks = {}
    for eval_v in ["v1_baseline", best_v]:
        if eval_v in bias_checks: continue
        template_str = VERSIONS[eval_v]
        orig_exp = np.array(results[eval_v]["expected_scores"])
        orig_round = np.array(results[eval_v]["rounded_scores"])
        
        swapped_exp = []
        swapped_round = []
        for p in tuning_pairs:
            prompt = format_prompt(template_str, p["job_title"], p["job_desc"], p["cv_text"], tokenizer, swap_order=True)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids_tensor = torch.tensor([input_ids]).to(model.device)
            
            with torch.inference_mode():
                outputs = model(input_ids=input_ids_tensor)
                logits = outputs.logits[0, -1, :]
                exp_s, round_s, probs, p_mass = compute_expected_score(logits)
            swapped_exp.append(exp_s)
            swapped_round.append(round_s)
            
        swap_exp_arr = np.array(swapped_exp)
        swap_round_arr = np.array(swapped_round)
        
        signed_diff = float(np.mean(orig_exp - swap_exp_arr))
        flip_fraction = float(np.mean(orig_round != swap_round_arr))
        
        # Per-role-family agreement
        rf_records = {}
        for p, h, j_round, j_exp in zip(tuning_pairs, hand_labels, orig_round, orig_exp):
            rf = p["role_family"]
            if rf not in rf_records:
                rf_records[rf] = {"hand": [], "judge_round": [], "judge_exp": []}
            rf_records[rf]["hand"].append(h)
            rf_records[rf]["judge_round"].append(j_round)
            rf_records[rf]["judge_exp"].append(j_exp)
            
        rf_summary = {}
        for rf, d in sorted(rf_records.items()):
            h_arr = np.array(d["hand"])
            jr_arr = np.array(d["judge_round"])
            je_arr = np.array(d["judge_exp"])
            rf_summary[rf] = {
                "n": len(h_arr),
                "exact_agreement": float(np.mean(h_arr == jr_arr)),
                "hand_mean": float(np.mean(h_arr)),
                "judge_mean": float(np.mean(je_arr))
            }
            
        bias_checks[eval_v] = {
            "order_swap": {
                "mean_signed_diff": signed_diff,
                "flip_fraction": flip_fraction,
                "orig_mean": float(np.mean(orig_exp)),
                "swap_mean": float(np.mean(swap_exp_arr))
            },
            "role_family_agreement": rf_summary
        }

    # Summary payload
    summary = {
        "best_version": best_v,
        "best_qwk": best_qwk,
        "best_template_hash": results[best_v]["template_hash"],
        "hand_cv_length_spearman": {
            "tokens": float(hand_cv_tok_spearman),
            "chars": float(hand_cv_char_spearman)
        },
        "versions": results,
        "bias_checks": bias_checks
    }

    with open("data/processed/prompt_tuning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save frozen best prompt
    frozen_prompt_path = "data/processed/frozen_judge_prompt_template.txt"
    with open(frozen_prompt_path, "w") as f:
        f.write(VERSIONS[best_v])
        
    print(f"\nFrozen best prompt template saved to {frozen_prompt_path}")
    print(f"Results saved to data/processed/prompt_tuning_summary.json")

if __name__ == "__main__":
    main()
