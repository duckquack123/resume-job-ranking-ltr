#!/bin/bash
#SBATCH --job-name=order_swap
#SBATCH --partition=mtech
#SBATCH --nodelist=cn03
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=run_order_swap.out

source ~/miniconda3/bin/activate vllm-env
export HF_HOME=/scratch/m25csa007/huggingface
export HF_TOKEN=$(grep HF_TOKEN /csehome/m25csa007/m25csa007/my.env | awk -F' = ' '{print $2}')
python run_order_swap_check.py
