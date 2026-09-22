import argparse
import pandas as pd
import json
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Export jobs and pooled CVs to local JSON for annotation.")
    parser.add_argument("--pool_results", type=str, help="Path to pooling results JSONL or Parquet.")
    parser.add_argument("--job_split", type=str, help="Which job split to export (dry_run, pilot, main).")
    parser.add_argument("--output_dir", type=str, default="data/annotation_exports")
    return parser.parse_args()

def deduplicate_pooled_candidates(candidates):
    """
    Apply within-pool near-duplicate removal.
    Drop candidates that are near-duplicates of a higher-ranked candidate
    within the same job's pool, keeping only one.
    """
    pass

def load_data(job_split):
    """
    Load jobs, CVs, and pooling results for the given split.
    Verify against frozen job IDs.
    """
    pass

def generate_annotator_tasks(jobs, cv_pools):
    """
    Format tasks. Each annotator receives whole jobs (all candidates for a job).
    - Randomize candidate order per job.
    - Generate stable pair IDs.
    - Exclude partner labels and model provenance.
    """
    pass

def export_provenance_mapping(tasks, output_dir):
    """
    Save the mapping between stable pair IDs and actual model provenance / ranking.
    This file is kept separate and NEVER shared with annotators.
    """
    pass

def save_json_tasks(tasks, output_dir, split_name):
    """
    Save the formatted tasks to local JSON files for offline annotation.
    Separate files depending on single-annotated vs double-annotated logic.
    """
    pass

def main():
    """
    Main execution flow (currently a skeleton).
    """
    args = parse_args()
    print("Annotation export skeleton running...")
    # data = load_data(args.job_split)
    # processed_pools = deduplicate_pooled_candidates(data['candidates'])
    # tasks = generate_annotator_tasks(data['jobs'], processed_pools)
    # export_provenance_mapping(tasks, args.output_dir)
    # save_json_tasks(tasks, args.output_dir, args.job_split)

if __name__ == "__main__":
    main()
