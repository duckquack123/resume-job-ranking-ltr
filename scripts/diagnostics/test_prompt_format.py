import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token
    )
    model.eval()

    with open("data/processed/timing_tasks.json") as f:
        tasks = json.load(f)[:10]

    rubric_user = """You are an expert HR recruiter evaluating a candidate's CV against a Job Description.
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

    # We will test two formatting styles:
    # Style 1: Chat template + assistant turn prefix 'Grade (0-3): '
    # Style 2: Raw prompt ending with '\nGrade (0-3): '
    
    styles = ["chat_template", "raw_prompt"]

    for style in styles:
        print(f"\n==================== STYLE: {style} ====================")
        digit_masses = []
        for i, t in enumerate(tasks):
            title = t.get("job_title", "")
            desc = t.get("job_desc", "")
            cv = t.get("cv_text", "")
            user_content = rubric_user.format(title=title, desc=desc, cv=cv)

            if style == "chat_template":
                messages = [{"role": "user", "content": user_content}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Grade (0-3): "
            else:
                prompt = user_content + "\nGrade (0-3): "

            # Verify tokenization boundary
            enc_prompt = tokenizer.encode(prompt, add_special_tokens=(style == "raw_prompt"))
            for d, target_id in [("0", 15), ("1", 16), ("2", 17), ("3", 18)]:
                enc_with_d = tokenizer.encode(prompt + d, add_special_tokens=(style == "raw_prompt"))
                if enc_with_d != enc_prompt + [target_id]:
                    print(f"Boundary mismatch for {d}! Expected {enc_prompt + [target_id]}, got {enc_with_d}")

            input_ids = torch.tensor([enc_prompt]).to(model.device)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

            topk = torch.topk(probs, 5)
            top5_tokens = [tokenizer.decode([idx.item()]) for idx in topk.indices]
            top5_probs = [round(p.item(), 4) for p in topk.values]
            top5_repr = list(zip(top5_tokens, topk.indices.tolist(), top5_probs))

            p0 = probs[15].item()
            p1 = probs[16].item()
            p2 = probs[17].item()
            p3 = probs[18].item()
            digit_mass = p0 + p1 + p2 + p3
            digit_masses.append(digit_mass)

            print(f"Pair {i} ({t['pair_id']}):")
            print(f"  Top-5: {top5_repr}")
            print(f"  Digit mass: {digit_mass:.4f} (0: {p0:.4f}, 1: {p1:.4f}, 2: {p2:.4f}, 3: {p3:.4f})")

        print(f"Mean digit mass for {style}: {sum(digit_masses)/len(digit_masses):.4f}")

if __name__ == "__main__":
    main()
