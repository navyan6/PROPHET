#!/bin/bash
# Run this ON PARCC (login node) after cloning the repo and syncing data.
# It prepares result directories and copies the -ESM ablation FASTA from
# the existing Stage 1 outputs so run_ablations.py finds all required files.
#
# Usage: bash setup_parcc_results.sh
set -e

REPO=/vast/projects/pranam/lab/$USER/hadsbm-hiv
cd "$REPO"

echo "[INFO] Creating result directories..."
mkdir -p \
    results/hiv_prophet_final \
    results/stage1_ablations \
    results/stage1_tevo \
    results/stage1_j_sweep \
    results/ablations \
    logs

# -ESM ablation: use the unfiltered T=0.15 Gibbs variants (500 seqs, no ESM filter).
# This is already available from the main Stage 1 run.
SRC="results/hiv_prophet_final/hiv_prophet_t015_variants.fasta"
DST="results/stage1_ablations/hiv_no_esm_gibbs_variants.fasta"

if [ -f "$SRC" ]; then
    if [ ! -f "$DST" ]; then
        cp "$SRC" "$DST"
        echo "[INFO] Copied -ESM FASTA: $DST  ($(grep -c '>' "$DST") seqs)"
    else
        echo "[INFO] -ESM FASTA already exists: $DST"
    fi
else
    echo "[WARN] Source FASTA not found: $SRC"
    echo "       Run stage 1 first or sync results/hiv_prophet_final/ from local."
fi

echo ""
echo "=============================="
echo "Setup complete."
echo "Next steps:"
echo "  1. sbatch run_all_stage2.slurm          # Tables 2, 4 (CVaR), 5 (eta), 6"
echo "  2. sbatch run_stage1_tevo.slurm         # Table 5 (T_evo sweep)"
echo "  3. sbatch run_stage1_ablation.slurm     # Table 4 (-DCA, -lambda)"
echo "  4. sbatch run_stage1_j_sweep.slurm      # Table 7 (J=25, 50)"
echo "  After each stage1 sweep: resubmit run_stage2_tevo.slurm (for tevo)"
echo "  After j_sweep: re-run run_ablations.py to pick up J_SWEEP_DIRS FASTAs"
echo "=============================="
