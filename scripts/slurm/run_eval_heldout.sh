#!/bin/bash
#SBATCH --job-name=eval_heldout
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=eval_heldout.out

export HF_HOME="/scratch/m25csa007/huggingface"
source /csehome/m25csa007/miniconda3/bin/activate vllm-env

python -u src/eval_heldout.py
