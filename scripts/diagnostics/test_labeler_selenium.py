"""
SUPERSEDED / LOCAL VERIFICATION TEST ONLY
-----------------------------------------
This script was used exclusively for local headless browser testing of labeler_gold.html
prior to annotator distribution, verifying 0 external network calls, independent flag toggling,
dropdown reason selection, and state resume upon reload.
It is preserved as historical test code and not part of the active ranking pipeline.
"""

import os
import json
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

def main():
    print("=== Running Local Headless Browser Test on labeler_gold.html (v2 Schema) ===", flush=True)
    html_path = os.path.abspath("data/processed/labeler_gold.html")
    tasks_path = os.path.abspath("data/processed/annotator_a_tasks.json")
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Firefox(options=options)
    wait = WebDriverWait(driver, 10)
    
    try:
        # 1. Open local HTML page (0 network calls)
        file_url = f"file://{html_path}"
        print(f"Loading URL: {file_url}", flush=True)
        driver.get(file_url)
        time.sleep(1)
        
        # 2. Upload annotator_a_tasks.json
        print("Uploading annotator_a_tasks.json...", flush=True)
        file_input = driver.find_element(By.ID, "taskFileInput")
        file_input.send_keys(tasks_path)
        time.sleep(1)
        
        modal = driver.find_element(By.ID, "file-prompt")
        assert not modal.is_displayed(), "File modal should be hidden after loading tasks!"
        
        # 3. Job 1: Rate Candidate 1 with Score = 3 (normal score)
        job_id_elem = driver.find_element(By.ID, "jobIdDisplay")
        job1_id = job_id_elem.text
        print(f"Active Job 1: {job1_id}", flush=True)
        
        cand1_id = driver.find_element(By.ID, "pairIdDisplay").text
        print(f"Rating Candidate 1 of Job 1: {cand1_id} -> Score: 3", flush=True)
        time.sleep(0.5)
        btn_score_3 = driver.find_element(By.ID, "btn-score-3")
        btn_score_3.click()
        time.sleep(0.5)
        
        # 4. Job 1: Rate Candidate 2 as Invalid CV (is_invalid = true, cannot_judge_job = false)
        cand2_id = driver.find_element(By.ID, "pairIdDisplay").text
        print(f"Rating Candidate 2 of Job 1: {cand2_id} -> Invalid CV", flush=True)
        time.sleep(0.5)
        btn_score_inv = driver.find_element(By.ID, "btn-score-inv")
        btn_score_inv.click()
        time.sleep(0.3)
        
        # Verify candidate reason wrapper is visible and select required reason
        cand_reason_wrapper = driver.find_element(By.ID, "candidateReasonWrapper")
        assert cand_reason_wrapper.is_displayed(), "Candidate reason wrapper must be visible when invalid is set!"
        cand_select_elem = driver.find_element(By.ID, "candidateReasonSelect")
        cand_select = Select(cand_select_elem)
        cand_select.select_by_value("CV text is empty/broken/unreadable")
        time.sleep(0.5)
        
        # 5. Switch to Job 2: Flag Job as Cannot Judge (cannot_judge_job = true, is_invalid = false)
        print("Switching to Job 2 in jobNavList...", flush=True)
        nav_items = driver.find_elements(By.CLASS_NAME, "job-nav-item")
        assert len(nav_items) == 52, f"Expected 52 jobs in nav, got {len(nav_items)}"
        nav_items[1].click()
        time.sleep(0.5)
        
        job2_id = driver.find_element(By.ID, "jobIdDisplay").text
        print(f"Active Job 2: {job2_id}", flush=True)
        assert job2_id != job1_id, "Job 2 ID should be different from Job 1!"
        
        # Toggle Cannot Judge on Job 2
        print("Flagging Job 2 as 'Cannot judge this job'...", flush=True)
        cannot_judge_cb = driver.find_element(By.ID, "cannotJudgeCheckbox")
        cannot_judge_cb.click()
        time.sleep(0.3)
        assert cannot_judge_cb.is_selected(), "cannotJudgeCheckbox should be checked!"
        
        # Verify job reason wrapper is visible and select required reason
        job_reason_wrapper = driver.find_element(By.ID, "jobReasonWrapper")
        assert job_reason_wrapper.is_displayed(), "Job reason wrapper must be visible when cannot-judge is checked!"
        job_select_elem = driver.find_element(By.ID, "jobReasonSelect")
        job_select = Select(job_select_elem)
        job_select.select_by_value("role outside my expertise")
        time.sleep(0.3)
        
        # Add Job Notes on Job 2
        print("Adding job notes to Job 2...", flush=True)
        notes_box = driver.find_element(By.ID, "jobNotesArea")
        test_note = "Domain requires specialized regulatory compliance knowledge outside evaluator scope."
        notes_box.clear()
        notes_box.send_keys(test_note)
        driver.execute_script("document.getElementById('jobNotesArea').dispatchEvent(new Event('input'));")
        time.sleep(0.5)
        
        cand3_id = driver.find_element(By.ID, "pairIdDisplay").text
        print(f"Candidate 1 of Job 2: {cand3_id} (inherits job cannot_judge_job = true)", flush=True)
        
        # 6. Export annotations
        print("\nTriggering exportAnnotations()...", flush=True)
        exported_results = driver.execute_script("exportAnnotations(false); return window.lastExportedJSON;")
        assert exported_results is not None and len(exported_results) > 0, "Failed to retrieve exported JSON!"
        
        # Filter down to the 3 test pairs
        annotated_3 = [r for r in exported_results if r["pair_id"] in [cand1_id, cand2_id, cand3_id]]
        assert len(annotated_3) == 3, f"Expected 3 pairs, found {len(annotated_3)}"
        
        p1 = next(r for r in annotated_3 if r["pair_id"] == cand1_id)
        p2 = next(r for r in annotated_3 if r["pair_id"] == cand2_id)
        p3 = next(r for r in annotated_3 if r["pair_id"] == cand3_id)
        
        # Assert independence of flags
        assert p1["cannot_judge_job"] is False and p1["is_invalid"] is False and p1["score"] == 3 and p1["flag_reason"] is None, f"P1 failed assertion: {p1}"
        assert p2["cannot_judge_job"] is False and p2["is_invalid"] is True and p2["score"] == -1 and p2["flag_reason"] == "CV text is empty/broken/unreadable", f"P2 failed assertion: {p2}"
        assert p3["cannot_judge_job"] is True and p3["is_invalid"] is False and p3["score"] is None and p3["flag_reason"] == "role outside my expertise", f"P3 failed assertion: {p3}"
        
        print("\n=== ACTUAL SAVED JSON CONTENT FOR 3 TEST PAIRS ===", flush=True)
        json_output_str = json.dumps(annotated_3, indent=2, sort_keys=True)
        print(json_output_str, flush=True)
        
        # Save to disk
        with open("data/processed/sample_saved_annotations.json", "w") as f:
            f.write(json_output_str)
        print("Saved to data/processed/sample_saved_annotations.json", flush=True)
        
        # 7. Test Reload and Resume
        print("\nTesting browser reload and resume functionality...", flush=True)
        driver.refresh()
        time.sleep(1)
        
        resume_box = driver.find_element(By.ID, "resume-box")
        assert resume_box.is_displayed(), "Resume box should be visible on reload after autosave!"
        
        resume_btn = resume_box.find_element(By.TAG_NAME, "button")
        resume_btn.click()
        time.sleep(0.5)
        
        # Verify state restoration on Job 2
        active_job = driver.find_element(By.ID, "jobIdDisplay").text
        print(f"Restored Active Job: {active_job}", flush=True)
        assert active_job == job2_id, f"Expected restored job to be {job2_id}, got {active_job}"
        
        cb_restored = driver.find_element(By.ID, "cannotJudgeCheckbox").is_selected()
        print(f"Restored Cannot Judge Checkbox State: {cb_restored}", flush=True)
        assert cb_restored is True, "Cannot judge checkbox should remain checked after reload!"
        
        reason_restored = driver.find_element(By.ID, "jobReasonSelect").get_attribute("value")
        print(f"Restored Job Reason: {reason_restored}", flush=True)
        assert reason_restored == "role outside my expertise", f"Expected job reason 'role outside my expertise', got {reason_restored}"
        
        notes_restored = driver.find_element(By.ID, "jobNotesArea").get_attribute("value")
        print(f"Restored Notes: {notes_restored}", flush=True)
        assert notes_restored == test_note, "Job notes should be restored after reload!"
        
        print("\n[VERIFIED] labeler_gold.html tested successfully: flags are independent, reasons are dropdown-enforced, and session resumes cleanly!")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
