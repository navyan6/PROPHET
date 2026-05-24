#!/bin/bash
# Stage 2 for J=200 (Table 7 J-sensitivity sweep).
# Prerequisite: run_stage1_j200.slurm must have completed.
# Submit: sbatch --dependency=afterok:<s1_j200_job_id> run_stage2_j200.slurm
#
#SBATCH --job-name=hiv_s2_j200
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:B200:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hiv_s2_j200_%j.out
#SBATCH --error=logs/hiv_s2_j200_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/$USER/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
export HF_HOME=/vast/projects/pranam/lab/$USER/.cache/huggingface

cd "$REPO"
mkdir -p logs results/ablations

WT_SEQ="PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
VARIANTS="results/stage1_j_sweep/J_200/hiv_train_gibbs_variants.fasta"

if [ ! -f "$VARIANTS" ]; then
    echo "ERROR: $VARIANTS not found. Run run_stage1_j200.slurm first."
    exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================="
echo "Job: $SLURM_JOB_ID — Stage 2 J=200"
echo "Variants: $VARIANTS"
echo "=============================="

$PYTHON prophet/stage2.py \
    --variants-fasta  "$VARIANTS" \
    --wt-seq          "$WT_SEQ" \
    --device          cuda:0 \
    --dfm-device      cuda:0 \
    --n-designs       500 \
    --n-steps         200 \
    --peptide-length  10 \
    --beta            5.0 \
    --eta             0.1 \
    --tau-bind        7.5 \
    --peptiverse-normalization raw \
    --variant-limit   149 \
    --out-json        "results/ablations/t7_J_200-${STAMP}.json" \
    --verbose-sampling

echo "Stage 2 J=200 complete."
