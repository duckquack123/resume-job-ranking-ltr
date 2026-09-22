#!/bin/bash
#SBATCH --partition=mtech
#SBATCH --nodelist=cn02
#SBATCH --gres=gpu:1
#SBATCH --output=run_hermes.out
nvidia-smi
source /csehome/m25csa007/miniconda3/bin/activate vllm-env
export HF_HOME=/scratch/m25csa007/huggingface
export HF_TOKEN=$(grep HF_TOKEN /csehome/m25csa007/m25csa007/my.env | awk -F' = ' '{print $2}')
python src/llm_judge.py --model NousResearch/Hermes-3-Llama-3.1-8B --tasks data/processed/timing_tasks.json --output data/processed/timing_results.json
