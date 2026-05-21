#!/bin/bash
# Table 4: Stage-1 component ablations — 3 jobs via SLURM array.
# Produces variant FASTAs for use by run_all_stage2.slurm (run_ablations.py).
#
# Array indices:
#   0 → −DCA  (--ablate-zero-dca-couplings)
#   1 → −λ    (--ablate-flatten-lambda)
#   2 → −ESM  (copy of unfiltered T=0.15 variants — no GPU needed for this one)
#
# Submit: sbatch run_stage1_ablation.slurm
# After all finish: re-submit run_all_stage2.slurm so ablation FASTAs are found.
#
#SBATCH --job-name=hiv_s1_abl
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:B200:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/hiv_s1_abl_%a_%j.out
#SBATCH --error=logs/hiv_s1_abl_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/$USER/hadsbm-hiv
export HF_HOME=/vast/projects/pranam/lab/$USER/.cache/huggingface

module load miniconda3/25.5.1
source activate hadsbm

cd "$REPO"
mkdir -p results/stage1_ablations logs

# Ablation 0 = no DCA (h=J=0), ablation 1 = no lambda weighting
ABL_FLAGS=(
    "--ablate-zero-dca-couplings"
    "--ablate-flatten-lambda"
)
ABL_NAMES=("no_dca" "no_lambda")

ABL_FLAG=${ABL_FLAGS[$SLURM_ARRAY_TASK_ID]}
ABL_NAME=${ABL_NAMES[$SLURM_ARRAY_TASK_ID]}
PREFIX="hiv_${ABL_NAME}"
OUT_DIR="results/stage1_ablations"

echo "=============================="
echo "Job        : $SLURM_JOB_ID  (array=$SLURM_ARRAY_TASK_ID)"
echo "Ablation   : $ABL_NAME ($ABL_FLAG)"
echo "=============================="

python prophet/stage1.py \
    --tree          data/pre_stage1_split/trees/train/hiv_train_tree.nwk \
    --trees-file    data/pre_stage1_split/trees/train/hiv_bootstrap_trees.txt \
    --fasta         data/pre_stage1_split/alignments/train/hiv_train_aligned.fasta \
    --protein \
    --out-dir       "$OUT_DIR" \
    --prefix        "$PREFIX" \
    --n-bootstraps  1 \
    --sample-variants 500 \
    --burn-in       200 \
    --t-evo         0.15 \
    --energy-mode   paper_dca \
    --ensemble-mode \
    --esm-device    cuda:0 \
    --esm-model     facebook/esm2_t33_650M_UR50D \
    --seed          42 \
    $ABL_FLAG

# Rename output to the filename expected by run_ablations.py
mv -f "$OUT_DIR/${PREFIX}_gibbs_variants.fasta" \
      "$OUT_DIR/hiv_${ABL_NAME}_gibbs_variants.fasta" 2>/dev/null || true

echo "Ablation $ABL_NAME complete → $OUT_DIR/hiv_${ABL_NAME}_gibbs_variants.fasta"
