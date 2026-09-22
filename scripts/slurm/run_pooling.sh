#!/bin/bash
#SBATCH --job-name=pool_500
#SBATCH --partition=mtech
#SBATCH --nodelist=cn02
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=run_pooling.out

source ~/miniconda3/bin/activate vllm-env
python -u src/pool_training_candidates.py

