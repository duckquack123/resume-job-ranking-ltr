import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_bm25(text):
    if not isinstance(text, str): return []
    text = text.lower()
    tokens = re.findall(r'\b[a-z]{2,}\b', text)
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return tokens

def get_bm25_topk(query_tokens, bm25_model, k):
    scores = bm25_model.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return top_indices

def get_dense_topk(query, corpus_embeddings, model, k):
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    cos_scores = torch.nn.functional.cosine_similarity(query_emb, corpus_embeddings)
    top_indices = torch.topk(cos_scores, k=k).indices.cpu().numpy()
    return top_indices

def overlap(l1, l2):
    return len(set(l1).intersection(set(l2)))

def run_reanalysis(k=5):
    out_dir = "/csehome/m25csa007/m25csa007/rjm-ltr/data/processed"
    jobs_dev = pd.read_parquet(f"{out_dir}/jobs_dev.parquet").head(5)
    cvs_eval = pd.read_parquet(f"{out_dir}/cvs_eval.parquet").head(10000).reset_index(drop=True)
    
    cv_texts = cvs_eval['CV'].fillna("").tolist()
    cv_keywords = cvs_eval['Primary Keyword'].fillna("").str.lower().tolist()
    cv_titles = cvs_eval['Position'].fillna("").tolist()
    
    # 5. Truncation Effect
    from transformers import AutoTokenizer
    tokenizer_bge = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
    
    cv_lengths = [len(tokenizer_bge.encode(t, add_special_tokens=True)) for t in cv_texts]
    jobs_lengths = [len(tokenizer_bge.encode(str(t), add_special_tokens=True)) for t in jobs_dev['Long Description']]
    
    cv_trunc_frac = sum(1 for l in cv_lengths if l > 512) / len(cv_lengths)
    jobs_trunc_frac = sum(1 for l in jobs_lengths if l > 512) / len(jobs_lengths)
    
    print("=== 5. TRUNCATION EFFECT ===")
    print(f"CVs exceeding 512 tokens: {cv_trunc_frac:.1%} (Max: {max(cv_lengths)})")
    print(f"Jobs exceeding 512 tokens: {jobs_trunc_frac:.1%} (Max: {max(jobs_lengths)})")
    
    # Random baseline for keyword match
    jobs_keywords = jobs_dev['Primary Keyword'].fillna("").str.lower().tolist()
    chance_baseline = sum(cv_keywords.count(jk)/len(cv_keywords) for jk in jobs_keywords) / len(jobs_keywords)
    
    # Setup BM25 (Fix 4: Proper tokenization and stopwords)
    tokenized_corpus = [preprocess_bm25(doc) for doc in cv_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Setup Dense 
    bge_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    bge_corpus = bge_model.encode(cv_texts, convert_to_tensor=True, show_progress_bar=True)
    
    e5_model = SentenceTransformer('intfloat/e5-small-v2')
    e5_passages = ["passage: " + t for t in cv_texts]
    e5_corpus = e5_model.encode(e5_passages, convert_to_tensor=True, show_progress_bar=True)
    
    # 2. Self-Retrieval Sanity Test
    print("\n=== 2. SELF-RETRIEVAL SANITY TEST ===")
    test_cv_idx = 0
    cv_text = cv_texts[test_cv_idx]
    
    bm25_self = get_bm25_topk(preprocess_bm25(cv_text), bm25, 1)[0]
    bge_self = get_dense_topk("Represent this sentence for searching relevant passages: " + cv_text, bge_corpus, bge_model, 1)[0]
    e5_self = get_dense_topk("query: " + cv_text, e5_corpus, e5_model, 1)[0]
    
    print(f"BM25 Self-Retrieval Rank 1 matches ID: {bm25_self == test_cv_idx}")
    print(f"BGE Self-Retrieval Rank 1 matches ID: {bge_self == test_cv_idx}")
    print(f"E5 Self-Retrieval Rank 1 matches ID: {e5_self == test_cv_idx}")
    
    bm25_bge_overlap = []
    bm25_e5_overlap = []
    bge_e5_overlap = []
    
    match_bm25, match_bge, match_e5 = 0, 0, 0
    total_queries = len(jobs_dev)
    
    print("\n=== 6. DRY-RUN JOBS TOP-3 ===")
    
    for idx, job_row in jobs_dev.iterrows():
        # Title + Requirements approach for BM25
        job_title = str(job_row['Position'])
        full_desc = str(job_row['Long Description'])
        job_keyword = str(job_row['Primary Keyword']).lower().strip()
        
        # BM25 Query: job title + full desc words (could isolate requirements, but structure varies)
        bm25_q = preprocess_bm25(job_title + " " + full_desc)
        bm25_indices = get_bm25_topk(bm25_q, bm25, k)
        
        # BGE Query (Fix 3: BGE uses instruction prefix for query)
        bge_q = "Represent this sentence for searching relevant passages: " + full_desc
        bge_indices = get_dense_topk(bge_q, bge_corpus, bge_model, k)
        
        # E5 Query (Fix 3: E5 uses query: prefix)
        e5_q = "query: " + full_desc
        e5_indices = get_dense_topk(e5_q, e5_corpus, e5_model, k)
        
        bm25_bge_overlap.append(overlap(bm25_indices, bge_indices) / k)
        bm25_e5_overlap.append(overlap(bm25_indices, e5_indices) / k)
        bge_e5_overlap.append(overlap(bge_indices, e5_indices) / k)
        
        for c_idx in bm25_indices: match_bm25 += int(cv_keywords[c_idx] == job_keyword)
        for c_idx in bge_indices: match_bge += int(cv_keywords[c_idx] == job_keyword)
        for c_idx in e5_indices: match_e5 += int(cv_keywords[c_idx] == job_keyword)
        
        print(f"\n--- Job {idx+1}: {job_title} ---")
        
        print(f"  [BM25]")
        for i in bm25_indices[:3]:
            print(f"    -> {cv_titles[i]} | {cv_texts[i][:300].replace(chr(10), ' ')}...")
            
        print(f"  [BGE]")
        for i in bge_indices[:3]:
            print(f"    -> {cv_titles[i]} | {cv_texts[i][:300].replace(chr(10), ' ')}...")
            
        print(f"  [E5]")
        for i in e5_indices[:3]:
            print(f"    -> {cv_titles[i]} | {cv_texts[i][:300].replace(chr(10), ' ')}...")

    print("\n=== 1 & 8. PAIRWISE OVERLAP ===")
    print(f"BM25 & BGE Overlap: {np.mean(bm25_bge_overlap):.1%}")
    print(f"BM25 & E5  Overlap: {np.mean(bm25_e5_overlap):.1%}")
    print(f"BGE  & E5  Overlap: {np.mean(bge_e5_overlap):.1%}")
    
    print("\n=== 7. PRIMARY KEYWORD MATCH RATE ===")
    print(f"Chance Baseline : {chance_baseline:.1%}")
    print(f"BM25 Match Rate : {match_bm25 / (total_queries * k):.1%}")
    print(f"BGE  Match Rate : {match_bge / (total_queries * k):.1%}")
    print(f"E5   Match Rate : {match_e5 / (total_queries * k):.1%}")

if __name__ == "__main__":
    run_reanalysis(k=5)
