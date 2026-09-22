#!/bin/bash
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn07
#SBATCH --output=/csehome/m25csa007/m25csa007/rjm-ltr/dedup_5gram.log
#SBATCH --job-name=dedup5gram

/csehome/m25csa007/miniconda3/envs/myenv/bin/python -u /csehome/m25csa007/m25csa007/rjm-ltr/src/fulldata_dedup_5gram.py
