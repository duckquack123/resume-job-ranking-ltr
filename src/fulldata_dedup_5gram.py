"""Compute full-data near-duplicate drop counts using word 5-grams, 128 perms."""
import pandas as pd, re
from datasketch import MinHash, MinHashLSH
from datasets import load_dataset

def get_minhash_5gram(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    if not isinstance(text, str): text = ''
    text = text.lower()
    tokens = [t for t in re.split(r'\W+', text) if t]
    shingles = [' '.join(tokens[i:i+5]) for i in range(max(1, len(tokens)-4))]
    for s in set(shingles): m.update(s.encode('utf-8'))
    return m

def count_drops(df, col, thresholds, num_perm=128):
    print(f'  Hashing {len(df)} rows...', flush=True)
    hashes = {idx: get_minhash_5gram(row[col], num_perm) for idx, row in df.iterrows()}
    counts = {}
    for thr in thresholds:
        print(f'  Testing threshold {thr}...', flush=True)
        lsh = MinHashLSH(threshold=thr, num_perm=num_perm)
        for idx, h in hashes.items(): lsh.insert(idx, h)
        to_drop = set()
        for idx, h in hashes.items():
            if idx in to_drop: continue
            res = set(lsh.query(h)) - {idx}
            to_drop.update(res)
        counts[thr] = len(to_drop)
        print(f'  Thr {thr}: Dropped {counts[thr]} (Kept {len(df)-counts[thr]})', flush=True)
    return counts

print('Loading raw data...')
jobs_ds = load_dataset('lang-uk/recruitment-dataset-job-descriptions-english', split='train')
jobs = jobs_ds.to_pandas().dropna(subset=['Long Description']).reset_index(drop=True)
jobs = jobs.loc[:, ~jobs.columns.duplicated()]
jobs = jobs[jobs['Long Description_lang'] == 'en']
print(f'Jobs (EN): {len(jobs)}')

cvs_ds = load_dataset('lang-uk/recruitment-dataset-candidate-profiles-english', split='train')
cvs = cvs_ds.to_pandas().dropna(subset=['CV']).reset_index(drop=True)
cvs = cvs.loc[:, ~cvs.columns.duplicated()]
cvs = cvs[cvs['CV_lang'] == 'en']
print(f'CVs (EN): {len(cvs)}')

print('\nJobs 5-gram dedup (128 perms):')
jd = count_drops(jobs, 'Long Description', [0.7, 0.8, 0.9])

print('\nCVs 5-gram dedup (128 perms):')
cd = count_drops(cvs, 'CV', [0.7, 0.8, 0.9])

print('\n=== SUMMARY ===')
print('Jobs 5-gram (128 perms, field: Long Description):')
for t, c in jd.items(): print(f'  Thr {t}: Dropped {c} (Kept {len(jobs)-c})')
print('CVs 5-gram (128 perms, field: CV):')
for t, c in cd.items(): print(f'  Thr {t}: Dropped {c} (Kept {len(cvs)-c})')
