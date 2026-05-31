#!/bin/bash
# Run Stage 2 comparison only (Stage 1 outputs must already exist).
#
# Usage:
#   sbatch --export=TARGET=zika_ns3 run_stage2.slurm
#
#SBATCH --job-name=s2
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:B200:6
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/s2_%x_%j.out
#SBATCH --error=logs/s2_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

set -e

REPO=/vast/projects/pranam/lab/nnori/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
export PATH=/vast/home/n/nnori/.conda/envs/hadsbm/bin:$PATH
export HF_HOME=/vast/projects/pranam/lab/nnori/.cache/huggingface
export TORCH_HOME=/vast/projects/pranam/lab/nnori/.cache/torch
export XDG_CACHE_HOME=/vast/projects/pranam/lab/nnori/.cache
export PYTHONUNBUFFERED=1

cd "$REPO"

if [ -z "${TARGET:-}" ]; then
    echo "ERROR: TARGET not set. Use --export=TARGET=<name>"
    exit 1
fi

mkdir -p logs

STAGE2_OUT="results/$TARGET/comparison"
mkdir -p "$STAGE2_OUT"

echo "=============================="
echo "Stage 2 only: $TARGET"
echo "Job $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=============================="

$PYTHON prophet/run_comparison.py \
    --target  "$TARGET" \
    --out-dir "$STAGE2_OUT" \
    --gpus    1,2,3,4,5

echo "=============================="
echo "Stage 2 complete: $TARGET"
echo "Results: $STAGE2_OUT"
echo "Finished: $(date)"
echo "=============================="
