#!/bin/bash
# PROPHET Stage 2 ablations for SARS-CoV-2 Mpro (nsp5 / 3CLpro).
# Runs Table 2 baselines + Table 4 ablations + CVaR sweep + holdout scoring.
#
# Prerequisite: run_sars_mpro_stage1.slurm AND run_sars_mpro_stage1_ablations.slurm complete.
# Submit: sbatch --dependency=afterok:<s1_id>:<s1_abl_id> run_sars_mpro_ablations_all.slurm
#
#SBATCH --job-name=sars_mpro_abl
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:B200:8
#SBATCH --mem=384G
#SBATCH --time=36:00:00
#SBATCH --output=logs/sars_mpro_abl_%j.out
#SBATCH --error=logs/sars_mpro_abl_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/nnori/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
export PATH=/vast/home/n/nnori/.conda/envs/hadsbm/bin:$PATH
export HF_HOME=/vast/projects/pranam/lab/nnori/.cache/huggingface

set -e
cd "$REPO"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs results/sars_mpro_ablations results/sars_mpro_holdout_scores

FASTA=data/sars_mpro/alignments/train/sars_mpro_train_aligned.fasta
HOLDOUT=data/sars_mpro/alignments/test/sars_mpro_test_clade_holdout.fasta
UNFILTERED=results/sars_mpro_prophet/sars_mpro_prophet_gibbs_variants.fasta

echo "=============================="
echo "Job $SLURM_JOB_ID — SARS Mpro Stage 2 ablations"
echo "=============================="

VARIANTS=$UNFILTERED
echo "Variants FASTA: $VARIANTS"

N_VARIANTS=$(grep -c "^>" "$VARIANTS" 2>/dev/null || true)
N_VARIANTS=${N_VARIANTS:-0}
VLIMIT=$(( N_VARIANTS < 149 ? N_VARIANTS : 149 ))
echo "Using $N_VARIANTS variants (limit=$VLIMIT)"

# WT consensus from training alignment (gap-fraction filter)
WT_SEQ=$($PYTHON -c "
from Bio import SeqIO
from collections import Counter
train = list(SeqIO.parse('$FASTA', 'fasta'))
aln_len = len(str(train[0].seq))
wt = ''
for i in range(aln_len):
    col = [str(r.seq)[i] for r in train]
    non_gap = [c for c in col if c != '-']
    if len(non_gap) / len(col) >= 0.5:
        wt += Counter(non_gap).most_common(1)[0][0]
print(wt)
")
echo "WT consensus length: ${#WT_SEQ}"

needs_run() {
    compgen -G "results/sars_mpro_ablations/${1}-*.json" > /dev/null 2>&1 && return 1 || return 0
}

run_exp() {
    local name=$1 fasta=$2 gpu=$3
    shift 3
    if ! needs_run "$name"; then
        echo "[skip] $name already has results"
        return
    fi
    if [ ! -f "$fasta" ]; then
        echo "[skip] $name: FASTA not found: $fasta"
        return
    fi
    local n_vars
    n_vars=$(grep -c "^>" "$fasta" 2>/dev/null || true)
    n_vars=${n_vars:-0}
    [ "$n_vars" -eq 0 ] && { echo "[skip] $name: empty FASTA"; return; }
    echo "Launching $name on GPU $gpu ($n_vars variants) ..."
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
    $PYTHON prophet/stage2.py \
        --variants-fasta "$fasta" \
        --wt-seq         "$WT_SEQ" \
        --device         "cuda:0" \
        --dfm-device     "cuda:0" \
        --n-designs       500 \
        --n-steps         200 \
        --peptide-length  10 \
        --beta            5.0 \
        --eta             0.1 \
        --tau-bind        8.0 \
        --peptiverse-normalization raw \
        --variant-limit   "$VLIMIT" \
        --guidance-var-limit 50 \
        --verbose-sampling \
        --seed           42 \
        "$@" \
        --out-json "results/sars_mpro_ablations/${name}-${STAMP}.json" \
        > "results/sars_mpro_ablations/${name}-${STAMP}.log" 2>&1 &
}

# ── Batch 1: Table 2 modes + Table 4 ablations ───────────────────────────────
echo "--- Batch 1: prophet / wt_only / random_variants / esm_only / leaves / -ESM / -DCA / -lambda ---"
run_exp "t2_prophet"           "$VARIANTS"   0
run_exp "t2_wt_only"           "$VARIANTS"   1 --design-mode wt_only
run_exp "t2_random_variants"   "$VARIANTS"   2 --design-mode random_variants
run_exp "t2_esm_only_variants" "$VARIANTS"   3 --design-mode esm_only_variants
run_exp "t2_uniform_leaves"    "$VARIANTS"   4 --design-mode uniform_leaves --guidance-variants-fasta "$FASTA"
run_exp "t4_no_esm"            "$UNFILTERED" 5
run_exp "t4_no_dca"    "results/sars_mpro_stage1_ablations/sars_mpro_no_dca_gibbs_variants.fasta"    6
run_exp "t4_no_lambda" "results/sars_mpro_stage1_ablations/sars_mpro_no_lambda_gibbs_variants.fasta" 7
wait; echo "Batch 1 done."

# ── Batch 2: Stage 1 ablations continued ─────────────────────────────────────
echo "--- Batch 2: -Conservation / -Bootstrap ---"
run_exp "t4_no_conserv"   "results/sars_mpro_stage1_ablations/sars_mpro_no_conserv_gibbs_variants.fasta"   0
run_exp "t4_no_bootstrap" "results/sars_mpro_stage1_ablations/sars_mpro_no_bootstrap_gibbs_variants.fasta" 1
wait; echo "Batch 2 done."

# ── Batch 3: CVaR η sweep ─────────────────────────────────────────────────────
echo "--- Batch 3: CVaR eta sweep (0.1, 0.5, 1.0) ---"
run_exp "t5_eta_01"  "$VARIANTS" 0 --eta 0.1
run_exp "t5_eta_05"  "$VARIANTS" 1 --eta 0.5
run_exp "t5_eta_10"  "$VARIANTS" 2 --eta 1.0
wait; echo "Batch 3 done."

# ── Batch 4: M sensitivity sweep ─────────────────────────────────────────────
echo "--- Batch 4: M sweep (variant-limit sensitivity) ---"
gpu=0
for M in 50 100 250 500; do
    if [ "$M" -le "$VLIMIT" ]; then
        MFASTA="$VARIANTS"
    else
        MFASTA="$UNFILTERED"
    fi
    run_exp "t6_M_${M}" "$MFASTA" $gpu --variant-limit $M
    gpu=$(( gpu + 1 ))
done
wait; echo "M sweep done."

echo ""
echo "All Stage 2 experiments complete."

# ── Phase 3: Holdout robustness scoring ───────────────────────────────────────
echo "--- Phase 3: Holdout robustness scoring ---"
if [ ! -f "$HOLDOUT" ]; then
    echo "WARNING: holdout FASTA not found: $HOLDOUT — skipping"
else
    $PYTHON scripts/score_holdout_robustness.py \
        --ablations-dir results/sars_mpro_ablations \
        --holdout-fasta "$HOLDOUT" \
        --out-dir       results/sars_mpro_holdout_scores \
        --device        cuda:0 \
        --tau           8.0
fi

echo "=============================="
echo "SARS Mpro pipeline complete."
echo "Stage 2 results: results/sars_mpro_ablations/"
echo "Holdout scores:  results/sars_mpro_holdout_scores/"
echo "=============================="
