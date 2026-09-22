import os
import re
import json
import hashlib

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    print("=== Step C Verification & Headless Validation ===")
    
    # 1. Hashes verification
    files_to_check = [
        "data/processed/annotator_b_job_ids.txt",
        "data/processed/gold_provenance.json",
        "data/processed/annotator_a_tasks.json",
        "data/processed/annotator_b_tasks.json",
        "data/processed/labeler_gold.html",
        "data/processed/dev_hand_labeling_tasks.json",
        "data/processed/frozen_feature_list.json",
        "data/processed/frozen_lgbm_config.json",
        "models/lambdamart_step1_frozen.txt"
    ]
    
    hashes = {}
    print("\n1. Computed SHA-256 Hashes:")
    for path in files_to_check:
        h = sha256_file(path)
        hashes[path] = h
        print(f"  {h}  {path}")
        
    # Check that dev_hand_labeling_tasks.json was NEVER modified
    expected_dev_hash = "2e6a5630e7b46e0e46257e58f7285d45cd5970497a67afd92e2800d14b449190"
    assert hashes["data/processed/dev_hand_labeling_tasks.json"] == expected_dev_hash, (
        f"CRITICAL ERROR: dev_hand_labeling_tasks.json was modified! Expected {expected_dev_hash}, got {hashes['data/processed/dev_hand_labeling_tasks.json']}"
    )
    print("\n[VERIFIED] dev_hand_labeling_tasks.json is 100% UNTOUCHED and matches frozen hash exactly.")
    
    # 2. Structural checks on tasks
    with open("data/processed/annotator_a_tasks.json") as f:
        tasks_a = json.load(f)
    with open("data/processed/annotator_b_tasks.json") as f:
        tasks_b = json.load(f)
    with open("data/processed/gold_provenance.json") as f:
        provenance = json.load(f)
    with open("data/processed/annotator_b_job_ids.txt") as f:
        b_ids = [x.strip() for x in f if x.strip()]
        
    print("\n2. Dataset Structure Audit:")
    print(f"  Annotator A Jobs: {len(tasks_a)} (expected 52)")
    print(f"  Annotator B Jobs: {len(tasks_b)} (expected 13)")
    assert len(tasks_a) == 52, f"Expected 52 jobs for A, got {len(tasks_a)}"
    assert len(tasks_b) == 13, f"Expected 13 jobs for B, got {len(tasks_b)}"
    assert len(b_ids) == 13, f"Expected 13 IDs in b_ids, got {len(b_ids)}"
    
    # Pairs counts
    pairs_a = [c["pair_id"] for j in tasks_a for c in j["candidates"]]
    pairs_b = [c["pair_id"] for j in tasks_b for c in j["candidates"]]
    print(f"  Annotator A Pairs: {len(pairs_a)} (446 unique)")
    print(f"  Annotator B Pairs: {len(pairs_b)} (115 unique)")
    assert len(pairs_a) == 446
    assert len(pairs_b) == 115
    
    # Check that B's jobs are an exact subset of A's jobs
    a_jids = set(j["job_id"] for j in tasks_a)
    b_jids = set(j["job_id"] for j in tasks_b)
    assert b_jids.issubset(a_jids), "Annotator B contains jobs not in Annotator A!"
    assert b_jids == set(b_ids), "Annotator B tasks jobs do not match annotator_b_job_ids.txt!"
    print("  [VERIFIED] Annotator B jobs are strict subset of Annotator A jobs and match annotator_b_job_ids.txt.")
    
    # Check that provenance is NOT leaked to annotator files
    for t in tasks_a + tasks_b:
        assert "provenance" not in t, "Provenance leaked in job task object!"
        for c in t["candidates"]:
            assert "provenance" not in c, "Provenance leaked in candidate object!"
            assert "system" not in c, "System name leaked in candidate object!"
    print("  [VERIFIED] Annotator files contain ZERO provenance or retriever system signals.")
    
    # Check that provenance has entries for all 446 pairs
    assert len(provenance) == 446, f"Expected 446 provenance entries, found {len(provenance)}"
    print("  [VERIFIED] Gold provenance file covers all 446 pairs.")
    
    # 3. Headless UI & Network Audit on labeler_gold.html
    print("\n3. Headless & Security Audit of labeler_gold.html:")
    with open("data/processed/labeler_gold.html") as f:
        html_content = f.read()
        
    # Check network calls
    network_patterns = [
        r"https?://",
        r"//[a-zA-Z0-9_\-\.]+",
        r"\bfetch\s*\(",
        r"\bXMLHttpRequest\b",
        r"\bWebSocket\b",
        r"\bnavigator\.sendBeacon\b"
    ]
    
    network_matches = []
    for pat in network_patterns:
        matches = re.findall(pat, html_content)
        if matches:
            # Filter out xmlns or schema definitions if any
            real_matches = [m for m in matches if not m.startswith("http://www.w3.org")]
            if real_matches:
                network_matches.extend(real_matches)
                
    print(f"  External network calls found: {len(network_matches)}")
    assert len(network_matches) == 0, f"External network calls detected in labeler_gold.html: {network_matches}"
    print("  [VERIFIED] Zero external network calls (100% offline self-contained).")
    
    # Check UI Elements
    required_elements = [
        ("Button '3 - Strong'", "3 - Strong"),
        ("Button '2 - Good with minor gaps'", "2 - Good with minor gaps"),
        ("Button '1 - Weak'", "1 - Weak"),
        ("Button '0 - Not relevant'", "0 - Not relevant"),
        ("Button 'Invalid / Unreadable'", "Invalid / Unreadable"),
        ("Rubric section", "Evaluation Rubric"),
        ("Ignore rules section", "Ground Rules (Ignore)"),
        ("Job notes area", "jobNotesArea"),
        ("Cannot judge job checkbox", "cannotJudgeCheckbox"),
        ("Timestamps recording", "start_ts"),
        ("Autosave / LocalStorage", "localStorage"),
        ("Resume session function", "resumeSession")
    ]
    
    for label, text_or_id in required_elements:
        assert text_or_id in html_content, f"Missing required element in HTML: {label} ({text_or_id})"
        print(f"  [PASS] {label}")
        
    print("\n=== All Step C Validations Passed Perfectly! ===")
    
    # Save verification report
    report = {
        "status": "ALL_TESTS_PASSED",
        "hashes": hashes,
        "tasks_a_summary": {"jobs": len(tasks_a), "pairs": len(pairs_a)},
        "tasks_b_summary": {"jobs": len(tasks_b), "pairs": len(pairs_b)},
        "offline_security_audit": "0 network calls, self-contained",
        "rubric_and_buttons_verified": True
    }
    with open("data/processed/step_c_verification_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
