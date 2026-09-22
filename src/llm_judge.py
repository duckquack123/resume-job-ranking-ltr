import json
import os
import argparse
import time
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# The default model, but can be overridden
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Input tasks JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_batch_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()

def build_prompt(title, desc, cv, tokenizer=None):
    user_content = f"""You are an expert technical recruiter evaluating a candidate's CV against a Job Description.
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
    if tokenizer is not None:
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Grade (0-3): "
        return prompt
    else:
        return user_content + "\nGrade (0-3): "

def compute_expected_score(logits, tokenizer):
    """
    Extracts the probabilities for 0, 1, 2, 3 from the logits of the last token.
    logits: 1D tensor of shape (vocab_size,)
    """
    target_tokens = {'0': 15, '1': 16, '2': 17, '3': 18}
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

def main():
    args = parse_args()

    results_set = set()
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        results_set.add(data["pair_id"])
                    except json.JSONDecodeError:
                        pass

    with open(args.tasks, "r") as f:
        tasks = json.load(f)

    pending_tasks = [t for t in tasks if t["pair_id"] not in results_set]
    print(f"Total tasks: {len(tasks)}, Pending: {len(pending_tasks)}")

    if not pending_tasks:
        print("All tasks completed.")
        return

    print(f"Loading {args.model} via transformers...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()

    print("Building prompts and handling truncation...")
    
    task_data = []
    truncation_count = 0
    
    for t in pending_tasks:
        title = t.get("job_title", "")
        desc = t.get("job_desc", "")
        cv = t.get("cv_text", "")
        
        if len(cv.strip()) < 50:
            task_data.append({
                "task": t,
                "prompt": "",
                "input_ids": [],
                "length": 0,
                "invalid": True,
                "truncated": False
            })
            continue

        prompt = build_prompt(title, desc, cv, tokenizer=tokenizer)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        
        truncated = False
        if len(tokens) > args.max_tokens - 1:
            truncation_count += 1
            truncated = True
            excess_tokens = len(tokens) - (args.max_tokens - 50)
            cv_chars_to_remove = int(excess_tokens * 4) 
            truncated_cv = cv[:-cv_chars_to_remove] if cv_chars_to_remove < len(cv) else cv[:100]
            prompt = build_prompt(title, desc, truncated_cv, tokenizer=tokenizer)
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) > args.max_tokens - 1:
                tokens = tokens[:args.max_tokens - 1]

        task_data.append({
            "task": t,
            "prompt": prompt,
            "input_ids": tokens,
            "length": len(tokens),
            "invalid": False,
            "truncated": truncated
        })

    print(f"Truncated {truncation_count} CVs due to length.")
    
    task_data.sort(key=lambda x: x["length"])
    
    total_pairs = 0
    total_tokens = 0
    
    print("Running PyTorch inference...")
    start_time = time.time()
    
    out_f = open(args.output, "a")
    
    # Dynamic batching
    batches = []
    current_batch = []
    current_max_len = 0
    
    for item in task_data:
        if item["invalid"]:
            batches.append([item])
            continue
            
        proposed_max_len = max(current_max_len, item["length"])
        if len(current_batch) > 0 and (len(current_batch) >= args.batch_size or proposed_max_len * (len(current_batch) + 1) > args.max_batch_tokens):
            batches.append(current_batch)
            current_batch = [item]
            current_max_len = item["length"]
        else:
            current_batch.append(item)
            current_max_len = proposed_max_len
            
    if current_batch:
        batches.append(current_batch)
    
    with torch.inference_mode():
        for batch in batches:
            valid_batch = [b for b in batch if not b["invalid"]]
            
            if valid_batch:
                max_len = max(b["length"] for b in valid_batch)
                input_ids_tensors = []
                attention_masks = []
                
                for b in valid_batch:
                    pad_len = max_len - b["length"]
                    input_ids_tensors.append(b["input_ids"] + [tokenizer.pad_token_id] * pad_len)
                    attention_masks.append([1] * b["length"] + [0] * pad_len)
                    
                input_ids_pt = torch.tensor(input_ids_tensors).to(model.device)
                attn_mask_pt = torch.tensor(attention_masks).to(model.device)
                
                outputs = model(input_ids=input_ids_pt, attention_mask=attn_mask_pt)
                
                for j, b in enumerate(valid_batch):
                    last_idx = b["length"] - 1
                    logits = outputs.logits[j, last_idx, :]
                    
                    expected, rounded, probs, prob_mass = compute_expected_score(logits, tokenizer)
                    
                    res = {
                        "pair_id": b["task"]["pair_id"],
                        "job_title": b["task"].get("job_title", ""),
                        "expected_score": expected,
                        "rounded_grade": rounded,
                        "raw_probs": probs,
                        "prob_mass": prob_mass,
                        "low_prob_flag": prob_mass < 0.5,
                        "truncation_applied": b["truncated"],
                        "invalid_cv": False
                    }
                    out_f.write(json.dumps(res) + "\n")
                    out_f.flush()
                    total_pairs += 1
                    total_tokens += b["length"]
            
            invalid_batch = [b for b in batch if b["invalid"]]
            for b in invalid_batch:
                res = {
                    "pair_id": b["task"]["pair_id"],
                    "job_title": b["task"].get("job_title", ""),
                    "expected_score": None,
                    "rounded_grade": None,
                    "raw_probs": {},
                    "prob_mass": 0.0,
                    "low_prob_flag": False,
                    "truncation_applied": False,
                    "invalid_cv": True
                }
                out_f.write(json.dumps(res) + "\n")
                out_f.flush()
                total_pairs += 1

    out_f.close()
    
    elapsed = time.time() - start_time
    
    if elapsed > 0 and total_pairs > 0:
        print(f"Processed {total_pairs} pairs in {elapsed:.2f}s. ({total_pairs/elapsed:.2f} pairs/s, {total_tokens/elapsed:.2f} tokens/s)")
        print(f"Longest prompt seen: {max([b['length'] for b in task_data if not b['invalid']], default=0)} tokens")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak GPU memory: {peak_mem:.2f} GB")

if __name__ == "__main__":
    main()
