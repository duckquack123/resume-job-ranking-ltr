#!/bin/bash
#SBATCH --partition=mtech
#SBATCH --nodelist=cn02
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=run_timing.out
nvidia-smi
source /csehome/m25csa007/miniconda3/bin/activate vllm-env
export HF_HOME=/scratch/m25csa007/huggingface
export HF_TOKEN=$(grep HF_TOKEN /csehome/m25csa007/m25csa007/my.env | awk -F' = ' '{print $2}')
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('Available:', torch.cuda.is_available())"
python src/llm_judge.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --tasks data/processed/timing_tasks.json --output data/processed/timing_results.jsonl --batch_size 4
