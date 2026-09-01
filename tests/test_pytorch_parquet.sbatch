#!/bin/bash
#SBATCH --job-name=test_pt_pq
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/netscratch/%u/test_pt_pq_%j.out
#SBATCH --error=/netscratch/%u/test_pt_pq_%j.err

source ~/venv/torch_env/bin/activate
python3 -u ~/bachelor-project/tests/test_pytorch_parquet.py
