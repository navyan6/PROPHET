#!/bin/bash
#SBATCH --job-name=hiv_stage2
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:B200:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hiv_stage2_%j.out
#SBATCH --error=logs/hiv_stage2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

# ── Paths ─────────────────────────────────────────────────────────────────────
# Repo lives in the lab project dir on VAST storage
REPO=/vast/projects/pranam/lab/$USER/hadsbm-hiv

# HuggingFace cache → project dir (keeps ESM2-650M off home quota)
export HF_HOME=/vast/projects/pranam/lab/$USER/.cache/huggingface

module load miniconda3/25.5.1
source activate hadsbm

cd "$REPO"
mkdir -p logs results/hiv_prophet_final

echo "=============================="
echo "Job        : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Python     : $(which python)"
echo "=============================="

python prophet/stage2.py \
    --variants-fasta  results/hiv_prophet_final/hiv_prophet_t015_esm_filtered_d20.fasta \
    --wt-seq          PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF \
    --device          cuda:0 \
    --dfm-device      cuda:0 \
    --n-designs       500 \
    --n-steps         200 \
    --peptide-length  10 \
    --eta             0.1 \
    --beta            5.0 \
    --omega-grid      10 \
    --guidance-var-limit 50 \
    --tau-bind        7.0 \
    --out-json        results/hiv_prophet_final/stage2_designs_${SLURM_JOB_ID}.json \
    --verbose-sampling

echo "=============================="
echo "Stage 2 complete."
echo "Scoring designs against holdout variants..."
echo "=============================="

# Score final designs against held-out escape variants (same holdout used for RFdiffusion baseline)
python scripts/evaluate_rf_diffusion_binding.py \
    --input           results/hiv_prophet_final/stage2_designs_${SLURM_JOB_ID}.json \
    --input-format    json \
    --target-seq      PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF \
    --escape-fasta    data/pre_stage1_split/alignments/test/hiv_test_clade_holdout.fasta \
    --out-csv         results/hiv_prophet_final/stage2_binding_${SLURM_JOB_ID}.csv \
    --device          cuda:0 \
    --tau-bind        7.0 \
    --sort-by         mean_score

echo "=============================="
echo "All done. Results:"
echo "  Designs : results/hiv_prophet_final/stage2_designs_${SLURM_JOB_ID}.json"
echo "  Binding : results/hiv_prophet_final/stage2_binding_${SLURM_JOB_ID}.csv"
echo "=============================="
