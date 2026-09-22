#!/bin/bash
#SBATCH --job-name=verify_gold
#SBATCH --partition=mtech
#SBATCH --time=00:05:00
#SBATCH --output=run_verify.out

source ~/miniconda3/bin/activate vllm-env
python -u src/verify_gold_and_test.py
