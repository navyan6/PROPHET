#!/bin/bash
# Table 5: T_evo sensitivity sweep — runs 4 Stage 1 jobs in parallel via SLURM array.
# Submit: sbatch run_stage1_tevo.slurm
# After all 4 finish, run: sbatch run_stage2_tevo.slurm  (or add to run_all_stage2.slurm manually)
#
# Array indices map to T_evo values:
#   0 → 0.5   1 → 1.0   2 → 2.0   3 → 5.0
#SBATCH --job-name=hiv_s1_tevo
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:B200:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/hiv_s1_tevo_%a_%j.out
#SBATCH --error=logs/hiv_s1_tevo_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/$USER/hadsbm-hiv
export HF_HOME=/vast/projects/pranam/lab/$USER/.cache/huggingface

module load miniconda3/25.5.1
source activate hadsbm

cd "$REPO"

T_VALS=(0.5 1.0 2.0 5.0)
T=${T_VALS[$SLURM_ARRAY_TASK_ID]}
T_TAG="t${T/./}"   # e.g. 0.5 → t05, 5.0 → t50

OUT_DIR="results/stage1_tevo/${T_TAG}"
mkdir -p "$OUT_DIR" logs

echo "=============================="
echo "Job        : $SLURM_JOB_ID  (array=$SLURM_ARRAY_TASK_ID)"
echo "T_evo      : $T"
echo "Out        : $OUT_DIR"
echo "=============================="

python prophet/stage1.py \
    --tree          data/pre_stage1_split/trees/train/hiv_train_tree.nwk \
    --trees-file    data/pre_stage1_split/trees/train/hiv_bootstrap_trees.txt \
    --fasta         data/pre_stage1_split/alignments/train/hiv_train_aligned.fasta \
    --protein \
    --out-dir       "$OUT_DIR" \
    --prefix        hiv_tevo_${T_TAG} \
    --n-bootstraps  1 \
    --sample-variants 500 \
    --burn-in       200 \
    --t-evo         "$T" \
    --energy-mode   paper_dca \
    --ensemble-mode \
    --esm-device    cuda:0 \
    --esm-model     facebook/esm2_t33_650M_UR50D \
    --seed          42

echo "Stage 1 T_evo=$T complete → $OUT_DIR"
