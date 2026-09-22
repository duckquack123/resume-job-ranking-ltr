import pandas as pd
import numpy as np
import random
import json
import hashlib
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch

out_dir = "/csehome/m25csa007/m25csa007/rjm-ltr/data/processed"

def get_jaccard(text1, text2):
    def get_shingles(t):
        if not isinstance(t, str): return set()
        words = [w for w in re.split(r'\W+', str(t).lower()) if w]
        return set([' '.join(words[i:i+5]) for i in range(max(1, len(words)-4))])
    
    s1 = get_shingles(text1)
    s2 = get_shingles(text2)
    if not s1 and not s2: return 0.0
    if not s1 or not s2: return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def build_labeler_html():
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Local CV Labeler</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }
        .container { display: flex; height: 90vh; gap: 20px; }
        .pane { flex: 1; padding: 20px; background: white; border-radius: 8px; overflow-y: auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .controls { margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px; }
        .job-title { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .score-btn { padding: 10px 20px; margin: 5px; cursor: pointer; border: 1px solid #ccc; background: #fff; border-radius: 4px; font-size: 16px; }
        .score-btn:hover { background: #e0e0e0; }
        pre { white-space: pre-wrap; font-family: inherit; }
    </style>
</head>
<body>
    <div class="controls">
        <input type="file" id="fileInput" accept=".json">
        <span id="progress" style="margin-left: 20px; font-weight: bold;"></span>
        <button id="saveBtn" style="float: right; padding: 10px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">Save Annotations</button>
    </div>
    
    <div class="container" id="workspace" style="display: none;">
        <div class="pane">
            <h2 class="job-title" id="jobTitle"></h2>
            <div id="jobDesc"></div>
        </div>
        <div class="pane">
            <div style="margin-bottom: 15px; background: #e8f4f8; padding: 15px; border-radius: 8px; text-align: center;">
                <button class="score-btn" onclick="score(3)">3 - Highly Relevant</button>
                <button class="score-btn" onclick="score(2)">2 - Somewhat Relevant</button>
                <button class="score-btn" onclick="score(1)">1 - Marginally Relevant</button>
                <button class="score-btn" onclick="score(0)">0 - Irrelevant</button>
                <button class="score-btn" onclick="score(-1)" style="background: #ffebee;">Invalid / Unreadable</button>
            </div>
            <hr>
            <div id="cvText" style="white-space: pre-wrap;"></div>
        </div>
    </div>

    <script>
        let tasks = [];
        let currentIndex = 0;
        let results = [];
        let startTime = null;

        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = function(event) {
                tasks = JSON.parse(event.target.result);
                results = [];
                currentIndex = 0;
                document.getElementById('workspace').style.display = 'flex';
                showTask();
            };
            reader.readAsText(file);
        });

        function showTask() {
            if (currentIndex >= tasks.length) {
                document.getElementById('workspace').innerHTML = '<h2>All tasks completed! Click Save Annotations.</h2>';
                return;
            }
            const task = tasks[currentIndex];
            document.getElementById('progress').innerText = `Task ${currentIndex + 1} of ${tasks.length}`;
            document.getElementById('jobTitle').innerText = task.job_title;
            document.getElementById('jobDesc').innerText = task.job_desc;
            document.getElementById('cvText').innerText = task.cv_text;
            startTime = Date.now();
        }

        function score(val) {
            const task = tasks[currentIndex];
            results.push({
                pair_id: task.pair_id,
                score: val,
                start_ts: startTime,
                end_ts: Date.now()
            });
            currentIndex++;
            showTask();
        }

        document.getElementById('saveBtn').addEventListener('click', function() {
            if (results.length === 0) return;
            const blob = new Blob([JSON.stringify(results, null, 2)], {type: 'application/json'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'annotations_dev.json';
            a.click();
        });
    </script>
</body>
</html>"""
    with open(f"{out_dir}/labeler.html", "w") as f:
        f.write(html)

def main():
    print("Loading data...")
    jobs_dev = pd.read_parquet(f"{out_dir}/jobs_dev.parquet")
    cvs_dev = pd.read_parquet(f"{out_dir}/cvs_dev.parquet")
    
    # 1. 10 Jobs (6 dry run + 4 pilot)
    with open(f"{out_dir}/dry_run_job_ids.txt", "r") as f:
        dry_run = [x.strip() for x in f if x.strip()]
    with open(f"{out_dir}/pilot_job_ids.txt", "r") as f:
        pilot = [x.strip() for x in f if x.strip()]
        
    random.seed(42)
    chosen_pilot = random.sample(pilot, 4)
    target_jobs = dry_run + chosen_pilot
    
    jobs = jobs_dev[jobs_dev['id'].isin(target_jobs)].copy()
    
    # 2. Exclude CVs
    with open(f"{out_dir}/excluded_cv_ids.txt", "r") as f:
        excluded = set(x.strip() for x in f if x.strip())
    cvs = cvs_dev[~cvs_dev['id'].isin(excluded)].reset_index(drop=True)
    
    # Encode CVs
    print("Encoding CVs...")
    cv_texts = cvs['CV'].fillna("").tolist()
    cv_ids = cvs['id'].tolist()
    
    tokenized_corpus = [str(doc).lower().split() for doc in cv_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bge_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    bge_corpus = bge_model.encode(cv_texts, convert_to_tensor=True, show_progress_bar=False)
    
    e5_model = SentenceTransformer('intfloat/e5-small-v2')
    e5_passages = ["passage: " + str(t) for t in cv_texts]
    e5_corpus = e5_model.encode(e5_passages, convert_to_tensor=True, show_progress_bar=False)
    
    K = 3
    tasks = []
    provenance = {}
    
    total_pairs = 0
    dupes_removed = 0
    
    random.seed(42) # For shuffling within tasks
    
    for _, job_row in jobs.iterrows():
        jid = job_row['id']
        jtitle = str(job_row['Position'])
        jdesc = str(job_row['Long Description'])
        
        # BM25
        query_tokens = str(jdesc).lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_idx = np.argsort(bm25_scores)[::-1][:K]
        
        # BGE
        bge_emb = bge_model.encode([jdesc], convert_to_tensor=True, show_progress_bar=False)
        bge_scores = torch.nn.functional.cosine_similarity(bge_emb, bge_corpus)
        bge_idx = torch.topk(bge_scores, k=K).indices.cpu().numpy()
        
        # E5
        e5_q = "query: " + jdesc
        e5_emb = e5_model.encode([e5_q], convert_to_tensor=True, show_progress_bar=False)
        e5_scores = torch.nn.functional.cosine_similarity(e5_emb, e5_corpus)
        e5_idx = torch.topk(e5_scores, k=K).indices.cpu().numpy()
        
        pool_cvs = {}
        pool_prov = {}
        
        def add_to_pool(indices, sys_name):
            for i in indices:
                cid = cv_ids[i]
                if cid not in pool_cvs:
                    pool_cvs[cid] = cv_texts[i]
                    pool_prov[cid] = []
                pool_prov[cid].append(sys_name)
                
        add_to_pool(bm25_idx, "BM25")
        add_to_pool(bge_idx, "BGE")
        add_to_pool(e5_idx, "E5")
        
        # Dedup within pool
        final_pool = []
        for cid, txt in pool_cvs.items():
            is_dupe = False
            for prev_cid in final_pool:
                prev_txt = pool_cvs[prev_cid]
                if get_jaccard(txt, prev_txt) >= 0.70:
                    is_dupe = True
                    break
            if is_dupe:
                dupes_removed += 1
            else:
                final_pool.append(cid)
                
        # Shuffle final pool
        random.shuffle(final_pool)
        
        job_tasks = []
        for cid in final_pool:
            pair_id = f"{jid}_{cid}"
            job_tasks.append({
                "pair_id": pair_id,
                "job_title": jtitle,
                "job_desc": jdesc,
                "cv_text": pool_cvs[cid],
                "company": str(job_row['Company Name']) # included for testing later, though labeler won't show it
            })
            provenance[pair_id] = pool_prov[cid]
            total_pairs += 1
            
        tasks.extend(job_tasks)

    # Save
    tasks_file = f"{out_dir}/dev_hand_labeling_tasks.json"
    with open(tasks_file, "w") as f:
        json.dump(tasks, f, indent=2)
        
    with open(f"{out_dir}/dev_hand_labeling_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
        
    build_labeler_html()
    
    with open(tasks_file, "rb") as f:
        tasks_hash = hashlib.sha256(f.read()).hexdigest()
        
    print(f"Total pairs generated: {total_pairs}")
    print(f"Per-job candidates (avg): {total_pairs / len(jobs)}")
    print(f"Within-pool duplicates removed: {dupes_removed}")
    print(f"Tasks file SHA-256: {tasks_hash}")
    
    # Print example
    print("\n--- Example Task ---")
    ex = tasks[0]
    print(f"Pair ID: {ex['pair_id']}")
    print(f"Job Title: {ex['job_title']}")
    print(f"CV Text (first 300 chars): {ex['cv_text'][:300]}...")
    
if __name__ == "__main__":
    main()
