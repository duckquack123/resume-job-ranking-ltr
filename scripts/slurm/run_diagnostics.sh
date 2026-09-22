#!/bin/bash
#SBATCH --job-name=diagnostics
#SBATCH --partition=mtech
#SBATCH --nodelist=cn02
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=run_diagnostics.out

source ~/miniconda3/bin/activate vllm-env
export HF_HOME=/scratch/m25csa007/huggingface
export HF_TOKEN=$(grep HF_TOKEN /csehome/m25csa007/m25csa007/my.env | awk -F' = ' '{print $2}')
python -u src/compute_tuning_diagnostics.py
