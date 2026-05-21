#!/bin/bash
# Table 5: Stage 2 for each T_evo variant set.
# Prerequisite: run_stage1_tevo.slurm must have completed for ALL 4 T values.
#
# Array indices:
#   0 → T=0.5   1 → T=1.0   2 → T=2.0   3 → T=5.0
#
# Submit AFTER all Stage 1 T_evo jobs finish:
#   sbatch --dependency=afterok:<stage1_tevo_job_id> run_stage2_tevo.slurm
#
#SBATCH --job-name=hiv_s2_tevo
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:B200:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/hiv_s2_tevo_%a_%j.out
#SBATCH --error=logs/hiv_s2_tevo_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/$USER/hadsbm-hiv
export HF_HOME=/vast/projects/pranam/lab/$USER/.cache/huggingface

module load miniconda3/25.5.1
source activate hadsbm

cd "$REPO"

T_VALS=(0.5 1.0 2.0 5.0)
T=${T_VALS[$SLURM_ARRAY_TASK_ID]}
T_TAG="t${T/./}"

VARIANTS="results/stage1_tevo/${T_TAG}/hiv_tevo_${T_TAG}_gibbs_variants.fasta"
OUT_DIR="results/ablations"
mkdir -p "$OUT_DIR" logs

WT_SEQ="PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

echo "=============================="
echo "Job        : $SLURM_JOB_ID  (array=$SLURM_ARRAY_TASK_ID)"
echo "T_evo      : $T"
echo "Variants   : $VARIANTS"
echo "=============================="

if [ ! -f "$VARIANTS" ]; then
    echo "ERROR: variant FASTA not found: $VARIANTS"
    echo "Run run_stage1_tevo.slurm first."
    exit 1
fi

python prophet/stage2.py \
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
    --out-json        "$OUT_DIR/t5_tevo_${T_TAG}-${SLURM_JOB_ID}.json" \
    --verbose-sampling

echo "Stage 2 T_evo=$T complete"
