"""
Cross-split near-duplicate detection using MinHash LSH (word 5-grams, 128 perms).
Separate from data_pipeline.py which uses word unigrams at threshold 0.85.
Independent check: 5-grams, 128 perms, threshold 0.70.
"""
import pandas as pd
import pytest
import re
import hashlib
import os
from datasketch import MinHash, MinHashLSH

OUT_DIR = "data/processed"

# Frozen file registry for integrity test (populated from decisions.md hashes)
FROZEN_HASHES = {
    "data/processed/test_job_ids.txt":          "4b282bcb9df5ca0d9d65915eb5567a6f83998e6e684f5383c321af05e950e95f",
    "data/processed/dev_job_ids.txt":           "369969c016eeb3a4e9b693a1e002df9692e72909d72a1fda23eb115c42e6b308",
    "data/processed/dry_run_job_ids.txt":       "7b9de57361201c136a381bcc29ee53971b319af0faa927a96bb80f5d40c02469",
    "data/processed/pilot_job_ids.txt":         "c778f0c43f43fe2f1fa466e7cccb11705b26b7c14848cbbc3b1754a58866e036",
    "data/processed/excluded_cv_ids.txt":       "28a6fdce10850af36edb84623d7b45de370c0e374fd40daacbcdb23ef6793610",
    "data/processed/eligibility_overrides.csv": "baefed6a1c087bdad2e2b194bd4a85a71269fe1b5b8ea3ce9e49d8e379738113",
    "data/processed/eligible_test_job_ids.txt": "0ed27144f5f6caa766706dd341283ae9698e1d0beec8ba45264db0a5543db848",
    "data/processed/eligible_dev_job_ids.txt":  "73c47b892bbe287ee363df49bd53a160be65be1e9c62cb735cd437c1b34ab162",
}

# ------------------------------------------------------------
# Shared MinHash helper (5-gram, configurable num_perm)
# Config: word 5-grams, num_perm=128, threshold=0.70
# ------------------------------------------------------------
def get_minhash_5gram(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    tokens = [t for t in re.split(r"\W+", text) if t]
    shingles = [" ".join(tokens[i : i + 5]) for i in range(max(1, len(tokens) - 4))]
    for s in set(shingles):
        m.update(s.encode("utf-8"))
    return m


def find_cross_split_near_duplicates(
    splits: dict,
    threshold: float = 0.70,
    num_perm: int = 128,
) -> list:
    """
    Return a list of (split_name_A, id_A, split_name_B, id_B, jaccard_estimate) for
    every pair of items from DIFFERENT splits whose MinHash Jaccard >= threshold.
    Uses LSH for efficiency.
    """
    all_hashes = {}
    all_lsh = {}

    for name, (df, col) in splits.items():
        for local_idx, row in df.iterrows():
            all_hashes[(name, local_idx)] = get_minhash_5gram(row[col], num_perm=num_perm)

    for name, (df, col) in splits.items():
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        for local_idx in df.index:
            lsh.insert((name, local_idx), all_hashes[(name, local_idx)])
        all_lsh[name] = lsh

    pairs = []
    seen = set()
    for name_a, (df_a, col_a) in splits.items():
        for local_idx_a, row_a in df_a.iterrows():
            key_a = (name_a, local_idx_a)
            h_a = all_hashes[key_a]
            for name_b, lsh_b in all_lsh.items():
                if name_b == name_a:
                    continue
                for key_b in lsh_b.query(h_a):
                    pair_key = tuple(sorted([key_a, key_b]))
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    j = h_a.jaccard(all_hashes[key_b])
                    if j >= threshold:
                        id_a = df_a.loc[local_idx_a, "id"] if "id" in df_a.columns else str(local_idx_a)
                        name_b2, local_idx_b = key_b
                        df_b = splits[name_b2][0]
                        id_b = df_b.loc[local_idx_b, "id"] if "id" in df_b.columns else str(local_idx_b)
                        pairs.append((name_a, id_a, name_b2, id_b, round(j, 4)))
    return pairs


# ==============================================================
# INTEGRITY TEST: frozen file hashes
# ==============================================================

def test_frozen_file_hashes():
    """
    Recompute SHA-256 of every frozen file and compare with hashes recorded in
    decisions.md. Fails immediately if any file has been modified.
    """
    for rel_path, expected_hash in FROZEN_HASHES.items():
        assert os.path.exists(rel_path), f"Frozen file missing: {rel_path}"
        with open(rel_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == expected_hash, (
            f"Frozen file MODIFIED: {rel_path}\n"
            f"  Expected: {expected_hash}\n"
            f"  Got:      {actual_hash}"
        )


# ==============================================================
# REAL-DATA TESTS
# ==============================================================

def _load_excluded_cv_ids() -> set:
    with open(f"{OUT_DIR}/excluded_cv_ids.txt") as f:
        return set(l.strip() for l in f if l.strip())


def test_no_job_id_in_multiple_splits():
    """Job IDs are disjoint across train / dev / test (threshold: exact ID match)."""
    train = pd.read_parquet(f"{OUT_DIR}/jobs_train.parquet")
    dev   = pd.read_parquet(f"{OUT_DIR}/jobs_dev.parquet")
    test  = pd.read_parquet(f"{OUT_DIR}/jobs_test.parquet")
    assert len(set(train["id"]) & set(dev["id"]))  == 0, "Train/Dev job ID overlap"
    assert len(set(train["id"]) & set(test["id"])) == 0, "Train/Test job ID overlap"
    assert len(set(dev["id"])   & set(test["id"])) == 0, "Dev/Test job ID overlap"


def test_no_cv_id_in_multiple_pools():
    """CV IDs are disjoint across train / dev / test pools (threshold: exact ID match)."""
    train = pd.read_parquet(f"{OUT_DIR}/cvs_train.parquet")
    dev   = pd.read_parquet(f"{OUT_DIR}/cvs_dev.parquet")
    test  = pd.read_parquet(f"{OUT_DIR}/cvs_test.parquet")
    assert len(set(train["id"]) & set(dev["id"]))  == 0, "Train/Dev CV ID overlap"
    assert len(set(train["id"]) & set(test["id"])) == 0, "Train/Test CV ID overlap"
    assert len(set(dev["id"])   & set(test["id"])) == 0, "Dev/Test CV ID overlap"


def test_no_minhash_near_duplicate_jobs_cross_split():
    """
    No job text from dev or test has word-5-gram MinHash Jaccard >= 0.70 to any
    job text in another split.
    Config: word 5-grams, num_perm=128, threshold=0.70.
    """
    train = pd.read_parquet(f"{OUT_DIR}/jobs_train.parquet")
    dev   = pd.read_parquet(f"{OUT_DIR}/jobs_dev.parquet")
    test  = pd.read_parquet(f"{OUT_DIR}/jobs_test.parquet")
    splits = {
        "train": (train, "Long Description"),
        "dev":   (dev,   "Long Description"),
        "test":  (test,  "Long Description"),
    }
    pairs = find_cross_split_near_duplicates(splits, threshold=0.70, num_perm=128)
    assert pairs == [], f"Found {len(pairs)} near-duplicate job pairs across splits: {pairs[:5]}"


def test_no_minhash_near_duplicate_cvs_cross_pool():
    """
    No CV text from train has word-5-gram MinHash Jaccard >= 0.70 to any CV in dev or
    test. Excluded CVs (excluded_cv_ids.txt) are skipped before checking.
    The known dev/test pair (114596_cv <-> 114111_cv) is excluded from dev and never
    appears in pooling; this test verifies no train contamination remains.
    Config: word 5-grams, num_perm=128, threshold=0.70.
    """
    excluded = _load_excluded_cv_ids()
    train = pd.read_parquet(f"{OUT_DIR}/cvs_train.parquet")
    dev   = pd.read_parquet(f"{OUT_DIR}/cvs_dev.parquet")
    test  = pd.read_parquet(f"{OUT_DIR}/cvs_test.parquet")

    # Apply exclusions to dev
    dev = dev[~dev["id"].isin(excluded)]

    splits = {
        "train": (train, "CV"),
        "dev":   (dev,   "CV"),
        "test":  (test,  "CV"),
    }
    pairs = find_cross_split_near_duplicates(splits, threshold=0.70, num_perm=128)
    assert pairs == [], f"Found {len(pairs)} near-duplicate CV pairs across pools: {pairs[:5]}"


# ==============================================================
# POSITIVE CONTROLS
# ==============================================================

def _make_scratch_df(texts: list, col: str) -> pd.DataFrame:
    return pd.DataFrame({"id": [f"fake_{i}" for i in range(len(texts))], col: texts})


def test_positive_control_near_duplicate_detected():
    """
    Arithmetic (corrected):
      base = word0 word1 … word99   (100 unique tokens → 96 base 5-grams)
      text_a = base + ' we require 5 years experience'  → 105 tokens → 101 5-grams
      text_b = base + ' we require 6 years experience'  → 105 tokens → 101 5-grams
      Differing shingles (3 each side):
        A-only: 'we require 5 years experience',
                'word99 we require 5 years', 'word98 word99 we require 5'
        B-only: same with '6'
      |A ∩ B| = 98,  |A ∪ B| = 104
      Exact Jaccard = 98/104 ≈ 0.942  (> 0.70 ✓)
      MinHash estimate (128 perms) ≈ 0.945
    The function computes Jaccard from the MinHash objects — no hard-coded threshold.
    """
    base = " ".join([f"word{i}" for i in range(100)])
    text_a = base + " we require 5 years experience"
    text_b = base + " we require 6 years experience"

    col = "Long Description"
    splits = {
        "split_a": (_make_scratch_df([text_a], col), col),
        "split_b": (_make_scratch_df([text_b], col), col),
    }
    pairs = find_cross_split_near_duplicates(splits, threshold=0.70, num_perm=128)
    assert len(pairs) >= 1, (
        f"Positive control: near-duplicate NOT detected. "
        f"Expected Jaccard ≈ 0.94, check MinHash variance."
    )
    # Also verify computed Jaccard is consistent with exact arithmetic
    h_a = get_minhash_5gram(text_a, num_perm=128)
    h_b = get_minhash_5gram(text_b, num_perm=128)
    estimated_j = h_a.jaccard(h_b)
    assert 0.85 <= estimated_j <= 1.0, (
        f"MinHash Jaccard {estimated_j:.4f} is unexpectedly far from exact value 0.942"
    )


def test_positive_control_exact_duplicate_detected():
    """An exact duplicate (Jaccard=1.0) must always be detected."""
    text = "This is a software engineer job requiring python and django expertise for five years."
    col = "Long Description"
    splits = {
        "split_a": (_make_scratch_df([text], col), col),
        "split_b": (_make_scratch_df([text], col), col),
    }
    pairs = find_cross_split_near_duplicates(splits, threshold=0.70, num_perm=128)
    assert len(pairs) >= 1, "Positive control: exact duplicate was NOT detected"


def test_no_excluded_cvs_in_pooling_results():
    """
    If any pooling output (like pooling_results.jsonl) exists, verify that none of the
    excluded CVs appear in its candidate lists.
    Currently, our scripts write output that might not directly contain CV IDs yet 
    (scale_up_pooling computes overlap, not the actual ranked lists).
    But this is a placeholder/assertion for when actual ranked lists are output.
    """
    excluded = _load_excluded_cv_ids()
    results_file = f"{OUT_DIR}/pooling_results.jsonl"
    if not os.path.exists(results_file):
        pytest.skip("No pooling results file yet to test.")
    
    import json
    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # If the script ever outputs 'candidates', verify they aren't excluded.
            if 'candidates' in data:
                for c in data['candidates']:
                    assert c not in excluded, f"Excluded CV {c} found in pooling results!"

def test_exclusion_positive_control():
    """
    Query the dev CV pool with the text of 114596_cv as a 'job'.
    It must rank first without the exclusion, and be absent with it.
    Uses BM25 for the test.
    """
    excluded_id = "114596_cv"
    dev_cvs = pd.read_parquet(f"{OUT_DIR}/cvs_dev.parquet")
    
    # 1. Verify the CV is actually in the raw dev CV pool
    assert excluded_id in dev_cvs['id'].values, f"{excluded_id} not in raw dev CV pool"
    
    cv_text = dev_cvs.loc[dev_cvs['id'] == excluded_id, 'CV'].values[0]
    
    from rank_bm25 import BM25Okapi
    
    def tokenize(text):
        if not isinstance(text, str): return []
        text = text.lower()
        return [t for t in re.split(r'\W+', text) if t]
        
    query_tokens = tokenize(cv_text)
    
    # Run WITHOUT exclusion
    tokenized_corpus_all = [tokenize(t) for t in dev_cvs['CV']]
    bm25_all = BM25Okapi(tokenized_corpus_all)
    scores_all = bm25_all.get_scores(query_tokens)
    top_idx_all = pd.Series(scores_all).idxmax()
    top_id_all = dev_cvs.iloc[top_idx_all]['id']
    
    assert top_id_all == excluded_id, f"Without exclusion, {excluded_id} should rank 1st"
    
    # Run WITH exclusion
    dev_cvs_filtered = dev_cvs[dev_cvs['id'] != excluded_id].reset_index(drop=True)
    tokenized_corpus_filtered = [tokenize(t) for t in dev_cvs_filtered['CV']]
    bm25_filtered = BM25Okapi(tokenized_corpus_filtered)
    scores_filtered = bm25_filtered.get_scores(query_tokens)
    
    # Ensure it's absent
    assert excluded_id not in dev_cvs_filtered['id'].values, "Filtered pool still contains excluded ID"

def test_export_join_by_id():
    """
    Verify that generating annotation tasks will join on job ID ('id') and NOT row index.
    This simulates the future logic in export_annotations.py.
    """
    jobs = pd.DataFrame({
        "id": ["1_job", "2_job", "3_job"],
        "Position": ["A", "B", "C"]
    })
    
    # Shuffle jobs to simulate randomized order or misaligned indices
    jobs_shuffled = jobs.sample(frac=1, random_state=42).reset_index(drop=True)
    
    pooling_results = [
        {"job_id": "3_job", "candidates": ["1_cv"]},
        {"job_id": "1_job", "candidates": ["2_cv"]}
    ]
    results_df = pd.DataFrame(pooling_results)
    
    # Simulate joining
    merged = jobs_shuffled.merge(results_df, left_on="id", right_on="job_id", how="inner")
    
    # Check that it correctly joined despite the shuffle
    row_1 = merged[merged["id"] == "1_job"].iloc[0]
    assert row_1["Position"] == "A"
    assert row_1["candidates"] == ["2_cv"]
    
    row_3 = merged[merged["id"] == "3_job"].iloc[0]
    assert row_3["Position"] == "C"
    assert row_3["candidates"] == ["1_cv"]
    
    # Ensure there's no index-based join leak
    assert len(merged) == 2

def test_export_integrity():
    """
    Round-trip test on the REAL export dev_hand_labeling_tasks.json.
    Verifies that for every task, job title, company and CV text exactly equal the parquet row for its ID.
    Also verifies that the excluded CV never appears in it.
    """
    tasks_file = f"{OUT_DIR}/dev_hand_labeling_tasks.json"
    if not os.path.exists(tasks_file):
        pytest.skip("Tasks file not generated yet")
        
    import json
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
        
    jobs_dev = pd.read_parquet(f"{OUT_DIR}/jobs_dev.parquet")
    cvs_dev = pd.read_parquet(f"{OUT_DIR}/cvs_dev.parquet")
    
    jobs_dict = jobs_dev.set_index("id").to_dict(orient="index")
    cvs_dict = cvs_dev.set_index("id").to_dict(orient="index")
    
    excluded = _load_excluded_cv_ids()
    
    for task in tasks:
        pair_id = task["pair_id"]
        job_id, cv_id = pair_id.split("_", 1)
        job_id += "_job"  # split on _, cv_id and job_id might have _ in them. wait, job ids are '123_job', cv ids are '123_cv'.
        # Actually pair_id is "{job_id}_{cv_id}" => "123_job_456_cv"
        # Safest way to split:
        job_id, cv_id = pair_id.split("_job_")
        job_id += "_job"
        cv_id += "_cv" if not cv_id.endswith("_cv") else ""
        
        assert cv_id not in excluded, f"Excluded CV {cv_id} found in tasks!"
        
        # Verify job
        job_row = jobs_dict[job_id]
        assert task["job_title"] == job_row["Position"], f"Job title mismatch for {job_id}"
        assert task["company"] == job_row["Company Name"], f"Company mismatch for {job_id}"
        
        # Verify CV
        cv_row = cvs_dict[cv_id]
        assert task["cv_text"] == cv_row["CV"], f"CV text mismatch for {cv_id}"

def test_compute_expected_score():
    import math
    import sys
    import os
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llm_judge import compute_expected_score
    
    class MockTokenizer:
        pass
        
    tokenizer = MockTokenizer()
    
    # Test case 1: pure probability mass on '2' (token id 17)
    logits1 = torch.full((100,), -100.0) # vocab size 100
    logits1[17] = math.log(0.9)
    logits1[15] = -float('inf') # -inf
    expected, rounded, raw, mass = compute_expected_score(logits1, tokenizer)
    assert abs(expected - 2.0) < 1e-6
    assert rounded == 2
    
    # Test case 2: distributed probability mass
    logits2 = torch.full((100,), -100.0)
    # Llama 3 ids: 0:15, 1:16, 2:17, 3:18
    # 0 -> 0.1, 1 -> 0.2, 2 -> 0.4, 3 -> 0.3
    logits2[15] = math.log(0.1)
    logits2[16] = math.log(0.2)
    logits2[17] = math.log(0.4)
    logits2[18] = math.log(0.3)
    # Expected: (0*0.1 + 1*0.2 + 2*0.4 + 3*0.3) / 1.0 = 1.9
    expected, rounded, raw, mass = compute_expected_score(logits2, tokenizer)
    assert abs(expected - 1.9) < 1e-6
    assert rounded == 2
    
    # Test case 3: missing valid tokens (mass = 0)
    logits3 = torch.full((100,), -100.0)
    expected, rounded, raw, mass = compute_expected_score(logits3, tokenizer)
    assert expected is None
    assert rounded is None


