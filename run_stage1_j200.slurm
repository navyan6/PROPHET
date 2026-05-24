#!/bin/bash
# Stage 1 for J=200 bootstrap trees (Table 7 J-sensitivity sweep).
# Prerequisite: run_build_bootstrap_200.slurm must have completed.
# Submit: sbatch --dependency=afterok:<boot200_job_id> run_stage1_j200.slurm
#
#SBATCH --job-name=hiv_s1_j200
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:B200:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/hiv_s1_j200_%j.out
#SBATCH --error=logs/hiv_s1_j200_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/$USER/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
export HF_HOME=/vast/projects/pranam/lab/$USER/.cache/huggingface

cd "$REPO"
mkdir -p logs results/stage1_j_sweep/J_200

echo "=============================="
echo "Job: $SLURM_JOB_ID — Stage 1 J=200"
echo "=============================="

$PYTHON prophet/stage1.py \
    --tree          data/pre_stage1_split/trees/train/hiv_train_tree.nwk \
    --trees-file    data/pre_stage1_split/trees/train/hiv_bootstrap_trees_200.txt \
    --fasta         data/pre_stage1_split/alignments/train/hiv_train_aligned.fasta \
    --protein \
    --out-dir       results/stage1_j_sweep/J_200 \
    --prefix        hiv_train \
    --n-bootstraps  1 \
    --sample-variants 500 \
    --burn-in       200 \
    --t-evo         0.15 \
    --energy-mode   paper_dca \
    --ensemble-mode \
    --esm-device    cuda:0 \
    --esm-model     facebook/esm2_t33_650M_UR50D \
    --esm-filter-delta 20 \
    --seed          42

echo "Stage 1 J=200 complete. Submit Stage 2: sbatch run_stage2_j200.slurm"
