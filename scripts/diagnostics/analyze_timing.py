import json
import numpy as np

def main():
    res_file = "data/processed/timing_results.jsonl"
    prob_masses = []
    expected_scores = []
    rounded_grades = []
    invalid_count = 0
    
    with open(res_file, "r") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            if d.get("invalid_cv", False):
                invalid_count += 1
            else:
                prob_masses.append(d["prob_mass"])
                expected_scores.append(d["expected_score"])
                rounded_grades.append(d["rounded_grade"])
                
    print(f"Total processed valid: {len(prob_masses)}")
    print(f"Invalid count: {invalid_count}")
    if len(prob_masses) > 0:
        print(f"Probability mass on digits - Mean: {np.mean(prob_masses):.4f}, Min: {np.min(prob_masses):.4f}, 5th Pct: {np.percentile(prob_masses, 5):.4f}")
        print(f"Expected scores - Mean: {np.mean(expected_scores):.4f}, Std: {np.std(expected_scores):.4f}, Min: {np.min(expected_scores):.2f}, Max: {np.max(expected_scores):.2f}")
        from collections import Counter
        print(f"Rounded grades distribution: {dict(Counter(rounded_grades))}")

if __name__ == "__main__":
    main()
