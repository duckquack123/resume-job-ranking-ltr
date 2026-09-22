import pandas as pd
import random

def main():
    # Load 500 job IDs
    with open("data/processed/train_500_job_ids.txt") as f:
        job_ids = [line.strip() for line in f if line.strip()]

    print(f"Total frozen 500-job set IDs: {len(job_ids)}")

    # Load jobs_train.parquet
    df_jobs = pd.read_parquet("data/processed/jobs_train.parquet")
    print(f"Total train jobs in parquet: {len(df_jobs)}")

    # Filter to the 500 jobs
    id_col = "id"
    df_500 = df_jobs[df_jobs[id_col].isin(job_ids)].copy()
    print(f"Matched 500 jobs: {len(df_500)}")

    # Role-family distribution
    role_col = "Role_Family"
    print("\n--- Role-Family Distribution ---")
    role_dist = df_500[role_col].value_counts()
    for role, count in role_dist.items():
        pct = count / len(df_500) * 100
        print(f"  {role}: {count} ({pct:.1f}%)")

    # Companies
    company_col = "Company Name"
    print("\n--- Company Statistics ---")
    num_unique_companies = df_500[company_col].nunique()
    max_per_company = df_500[company_col].value_counts().max()
    top_companies = df_500[company_col].value_counts().head(10)
    print(f"Unique companies count: {num_unique_companies}")
    print(f"Max jobs per company: {max_per_company}")
    print("Top companies counts:")
    for comp, count in top_companies.items():
        print(f"  {comp}: {count}")

    # 30 random titles (fixed seed 42)
    random.seed(42)
    title_col = "Position"
    all_titles = df_500[title_col].tolist()
    sample_30 = random.sample(all_titles, 30)
    print("\n--- 30 Random Titles (seed 42) ---")
    for i, t in enumerate(sample_30, 1):
        print(f"{i}. {t}")

    # Check near-duplicates within pool
    print("\n--- Within-pool Duplicate Check ---")
    exact_title_desc_dups = df_500.duplicated(subset=[title_col, "Long Description"]).sum()
    print(f"Exact title+desc duplicates: {exact_title_desc_dups}")
    exact_desc_dups = df_500.duplicated(subset=["Long Description"]).sum()
    print(f"Exact desc duplicates: {exact_desc_dups}")

if __name__ == "__main__":
    main()
