import pandas as pd
from collections import defaultdict
import numpy as np

out_dir = "/csehome/m25csa007/m25csa007/rjm-ltr/data/processed"

def analyze_role_family():
    print("=== ROLE FAMILY DISTRIBUTION ===")
    for split in ['train', 'dev', 'test']:
        df = pd.read_parquet(f"{out_dir}/jobs_{split}.parquet")
        counts = df['Role_Family'].value_counts()
        print(f"\n[{split.upper()}] Role Families:")
        print(counts.head(10))
        
        top_companies = df['Company Name'].value_counts().head(10)
        print(f"[{split.upper()}] Top Companies:")
        print(top_companies)

if __name__ == "__main__":
    analyze_role_family()
