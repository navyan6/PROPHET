#!/bin/bash
#SBATCH --job-name=hiv_all_stage2
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:B200:8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hiv_all_stage2_%j.out
#SBATCH --error=logs/hiv_all_stage2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/$USER/hadsbm-hiv
export HF_HOME=/vast/projects/pranam/lab/$USER/.cache/huggingface
export TORCH_HOME=/vast/projects/pranam/lab/$USER/.cache/torch

module load miniconda3/25.5.1
source activate hadsbm

cd "$REPO"
mkdir -p logs results/ablations

echo "=============================="
echo "Job        : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ×8"
echo "Python     : $(which python)"
echo "=============================="

# Launches all Stage 2 experiments (Tables 2, 4, 5, 6) in parallel across 8 GPUs.
# Stage 7 J-sweep FASTAs must exist first; script skips missing FASTAs gracefully.
python prophet/run_ablations.py

echo "=============================="
echo "All stage2 jobs launched — monitoring logs in results/ablations/"
echo "=============================="

# Wait for all background subprocesses to finish
wait
echo "All stage2 jobs complete."
