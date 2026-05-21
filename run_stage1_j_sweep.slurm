#!/bin/bash
# Table 7: Tree count J sensitivity sweep — Stage 1 re-runs at J=25 and J=50.
# J=100 is already done (our main run). J=200 requires 200 bootstrap trees;
# run build_trees with --n-bootstraps 200 first, then add J=200 here.
#
# Array indices:
#   0 → J=25
#   1 → J=50
#
# Submit: sbatch run_stage1_j_sweep.slurm
# After finish: update J_SWEEP_DIRS in prophet/run_ablations.py with the
#   variant FASTA paths and resubmit run_all_stage2.slurm.
#
#SBATCH --job-name=hiv_s1_jsweep
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:B200:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/hiv_s1_jsweep_%a_%j.out
#SBATCH --error=logs/hiv_s1_jsweep_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/$USER/hadsbm-hiv
export HF_HOME=/vast/projects/pranam/lab/$USER/.cache/huggingface

module load miniconda3/25.5.1
source activate hadsbm

cd "$REPO"

J_VALS=(25 50)
J=${J_VALS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="results/stage1_j_sweep/J_${J}"
mkdir -p "$OUT_DIR" logs

echo "=============================="
echo "Job        : $SLURM_JOB_ID  (array=$SLURM_ARRAY_TASK_ID)"
echo "J          : $J"
echo "Out        : $OUT_DIR"
echo "=============================="

python prophet/stage1.py \
    --tree               data/pre_stage1_split/trees/train/hiv_train_tree.nwk \
    --trees-file         data/pre_stage1_split/trees/train/hiv_bootstrap_trees.txt \
    --tree-subsample-j   "$J" \
    --tree-subsample-seed 42 \
    --fasta              data/pre_stage1_split/alignments/train/hiv_train_aligned.fasta \
    --protein \
    --out-dir            "$OUT_DIR" \
    --prefix             hiv_J${J} \
    --n-bootstraps       1 \
    --sample-variants    500 \
    --burn-in            200 \
    --t-evo              0.15 \
    --energy-mode        paper_dca \
    --ensemble-mode \
    --esm-device         cuda:0 \
    --esm-model          facebook/esm2_t33_650M_UR50D \
    --seed               42

# Rename to the filename expected by run_ablations.py J_SWEEP_DIRS
mv -f "$OUT_DIR/hiv_J${J}_gibbs_variants.fasta" \
      "$OUT_DIR/hiv_train_gibbs_variants.fasta" 2>/dev/null || true

echo "J=$J sweep complete → $OUT_DIR/hiv_train_gibbs_variants.fasta"
