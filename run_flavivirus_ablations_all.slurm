#!/bin/bash
# PROPHET Stage 2 ablations for flavivirus NS3 targets.
#
# Runs Table 2 experiments (prophet, wt_only, random_variants, uniform_leaves,
# esm_only_variants) plus Table 4 Stage 2 ablations (-ESM, CVaR η sweep)
# for all three viruses: dengue_ns3, zika_ns3, wnv_ns3.
#
# Prerequisite: run_flavivirus_stage1.slurm must have completed.
# Submit: sbatch --dependency=afterok:<stage1_job_id> run_flavivirus_ablations_all.slurm
#
#SBATCH --job-name=flavi_abl
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:B200:8
#SBATCH --mem=384G
#SBATCH --time=36:00:00
#SBATCH --output=logs/flavi_abl_%j.out
#SBATCH --error=logs/flavi_abl_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/nnori/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
export PATH=/vast/home/n/nnori/.conda/envs/hadsbm/bin:$PATH
export HF_HOME=/vast/projects/pranam/lab/nnori/.cache/huggingface

set -e
cd "$REPO"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs results/flavi_ablations results/flavi_holdout_scores

VIRUSES=(dengue_ns3 zika_ns3 wnv_ns3)

echo "=============================="
echo "Job $SLURM_JOB_ID — Flavivirus NS3 Stage 2 ablations"
echo "=============================="

# ── Helper: true if no JSON exists for this experiment tag ──────────────────
needs_run() {
    compgen -G "results/flavi_ablations/${1}-*.json" > /dev/null 2>&1 && return 1 || return 0
}

# ── Launch Stage 2 experiment ─────────────────────────────────────────────────
# Usage: run_exp <name> <variants_fasta> <gpu_id> <extra_args>
run_exp() {
    local name=$1 fasta=$2 gpu=$3
    shift 3
    if ! needs_run "$name"; then
        echo "[skip] $name already has results"
        return
    fi
    if [ ! -f "$fasta" ]; then
        echo "[skip] $name: variant FASTA not found: $fasta"
        return
    fi
    local n_vars
    n_vars=$(grep -c "^>" "$fasta" 2>/dev/null || true)
    n_vars=${n_vars:-0}
    if [ "$n_vars" -eq 0 ]; then
        echo "[skip] $name: empty variant FASTA"
        return
    fi
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
        --out-json "results/flavi_ablations/${name}-${STAMP}.json" \
        > "results/flavi_ablations/${name}-${STAMP}.log" 2>&1 &
}

# ── Per-virus: Stage 2 experiments ───────────────────────────────────────────
# Each batch of 3 viruses runs on 3 GPUs (0–2), with mode variants on 4–7.
# We run one experiment type at a time across all viruses, then wait.

# Collect per-virus metadata (WT consensus, variant paths, variant limits)
declare -A WT_SEQS VARIANTS UNFILTERED VLIMITS

for V in "${VIRUSES[@]}"; do
    FASTA="data/$V/alignments/train/${V}_train_aligned.fasta"
    OUT="results/flavi_prophet/$V"

    # WT = consensus using the same <=50% gap-fraction filter as Stage 1
    WT_SEQS[$V]=$($PYTHON -c "
from Bio import SeqIO
from collections import Counter
recs = list(SeqIO.parse('$FASTA', 'fasta'))
N = len(recs)
aln_len = len(str(recs[0].seq))
wt = ''
for i in range(aln_len):
    col = [str(r.seq)[i] for r in recs]
    non_gap = [c for c in col if c not in '-X']
    if (N - len(non_gap)) <= 0.50 * N:
        wt += Counter(non_gap).most_common(1)[0][0] if non_gap else 'A'
print(wt)
")

    # ESM-filtered preferred; fall back to raw Gibbs
    FILTERED=$(ls "$OUT/${V}_prophet_"*esm_filtered*.fasta 2>/dev/null | head -1)
    GIBBS="$OUT/${V}_prophet_gibbs_variants.fasta"
    VARIANTS[$V]="${FILTERED:-$GIBBS}"
    UNFILTERED[$V]="$GIBBS"

    N=$(grep -c "^>" "${VARIANTS[$V]}" 2>/dev/null || true)
    N=${N:-0}
    VLIMITS[$V]=$(( N < 149 ? N : 149 ))
    echo "  $V: WT=${#WT_SEQS[$V]}aa, ${N} variants (limit=${VLIMITS[$V]}), using ${VARIANTS[$V]##*/}"
done

# ── Batch 1: Table 2 modes — 5 experiments × 3 viruses = 15 tasks on 8 GPUs ──
# Round 1: 3 viruses × prophet mode (GPUs 0–2) + 3 viruses × wt_only (GPUs 3–5)
# Round 2: 3 viruses × random_variants (GPUs 0–2) + 3 viruses × esm_only (GPUs 3–5)
# Round 3: 3 viruses × uniform_leaves (GPUs 0–2) + 3 viruses × -ESM ablation (3–5)

echo ""
echo "--- Batch 1: Table 2 modes (Round 1: prophet + wt_only) ---"
gpu=0
for V in "${VIRUSES[@]}"; do
    WT_SEQ="${WT_SEQS[$V]}"
    VLIMIT="${VLIMITS[$V]}"
    run_exp "${V}_t2_prophet"  "${VARIANTS[$V]}" $gpu
    gpu=$(( gpu + 1 ))
done
for V in "${VIRUSES[@]}"; do
    WT_SEQ="${WT_SEQS[$V]}"
    VLIMIT="${VLIMITS[$V]}"
    run_exp "${V}_t2_wt_only"  "${VARIANTS[$V]}" $gpu --design-mode wt_only
    gpu=$(( gpu + 1 ))
done
wait; echo "Round 1 done."

echo "--- Batch 1 Round 2: random_variants + esm_only_variants ---"
gpu=0
for V in "${VIRUSES[@]}"; do
    WT_SEQ="${WT_SEQS[$V]}"
    VLIMIT="${VLIMITS[$V]}"
    run_exp "${V}_t2_random_variants"   "${VARIANTS[$V]}" $gpu --design-mode random_variants
    gpu=$(( gpu + 1 ))
done
for V in "${VIRUSES[@]}"; do
    WT_SEQ="${WT_SEQS[$V]}"
    VLIMIT="${VLIMITS[$V]}"
    run_exp "${V}_t2_esm_only_variants" "${VARIANTS[$V]}" $gpu --design-mode esm_only_variants
    gpu=$(( gpu + 1 ))
done
wait; echo "Round 2 done."

echo "--- Batch 1 Round 3: uniform_leaves + -ESM ablation ---"
gpu=0
for V in "${VIRUSES[@]}"; do
    WT_SEQ="${WT_SEQS[$V]}"
    VLIMIT="${VLIMITS[$V]}"
    TRAIN_FASTA="data/$V/alignments/train/${V}_train_aligned.fasta"
    run_exp "${V}_t2_uniform_leaves" "${VARIANTS[$V]}" $gpu \
        --design-mode uniform_leaves --guidance-variants-fasta "$TRAIN_FASTA"
    gpu=$(( gpu + 1 ))
done
for V in "${VIRUSES[@]}"; do
    WT_SEQ="${WT_SEQS[$V]}"
    VLIMIT="${VLIMITS[$V]}"
    run_exp "${V}_t4_no_esm" "${UNFILTERED[$V]}" $gpu
    gpu=$(( gpu + 1 ))
done
wait; echo "Round 3 done."

echo "--- Batch 2: CVaR η sweep ---"
gpu=0
for ETA in 0.1 0.5 1.0; do
    ETAG="${ETA/./}"
    for V in "${VIRUSES[@]}"; do
        WT_SEQ="${WT_SEQS[$V]}"
        VLIMIT="${VLIMITS[$V]}"
        run_exp "${V}_t5_eta_${ETAG}" "${VARIANTS[$V]}" $gpu --eta $ETA
        gpu=$(( gpu + 1 ))
        # reset GPU counter after 8
        [ $gpu -ge 8 ] && { wait; gpu=0; }
    done
done
wait; echo "CVaR sweep done."

echo "--- Batch 5: M sweep (variant-limit sensitivity) ---"
gpu=0
for M in 50 100 250 500; do
    for V in "${VIRUSES[@]}"; do
        WT_SEQ="${WT_SEQS[$V]}"
        VLIMIT="${VLIMITS[$V]}"
        if [ "$M" -le "$VLIMIT" ]; then
            MFASTA="${VARIANTS[$V]}"
        else
            MFASTA="${UNFILTERED[$V]}"
        fi
        run_exp "${V}_t6_M_${M}" "$MFASTA" $gpu --variant-limit $M
        gpu=$(( gpu + 1 ))
        [ $gpu -ge 8 ] && { wait; gpu=0; }
    done
done
wait; echo "M sweep done."

# ── Batch 3: Stage 1 ablations → Stage 2 (-DCA, -λ, -Conservation, -Bootstrap) ──
echo "--- Batch 3: Stage 1 ablation variants → Stage 2 ---"
gpu=0
for ABL in no_dca no_lambda no_conserv no_bootstrap; do
    ABL_TAG="t4_${ABL}"
    for V in "${VIRUSES[@]}"; do
        WT_SEQ="${WT_SEQS[$V]}"
        VLIMIT="${VLIMITS[$V]}"
        ABL_DIR="results/flavi_stage1_ablations/$V"

        # Prefer ESM-filtered from ablation; fall back to Gibbs
        ABL_FILTERED=$(ls "$ABL_DIR/${V}_${ABL}_"*esm_filtered*.fasta 2>/dev/null | head -1)
        ABL_GIBBS="$ABL_DIR/${V}_${ABL}_gibbs_variants.fasta"
        ABL_FASTA="${ABL_FILTERED:-$ABL_GIBBS}"

        run_exp "${V}_${ABL_TAG}" "$ABL_FASTA" $gpu
        gpu=$(( gpu + 1 ))
        [ $gpu -ge 8 ] && { wait; gpu=0; }
    done
    wait; gpu=0
done
wait; echo "Stage 1 ablations → Stage 2 done."

echo ""
echo "All Stage 2 experiments complete."

# ── Phase 3: Holdout robustness scoring ───────────────────────────────────────
echo "--- Phase 3: Holdout robustness scoring ---"

for V in "${VIRUSES[@]}"; do
    HOLDOUT="data/$V/alignments/test/${V}_test_holdout.fasta"
    if [ ! -f "$HOLDOUT" ]; then
        echo "WARNING: holdout FASTA not found for $V: $HOLDOUT — skipping"
        continue
    fi
    echo "Scoring $V holdout ..."
    $PYTHON scripts/score_holdout_robustness.py \
        --ablations-dir results/flavi_ablations \
        --holdout-fasta "$HOLDOUT" \
        --out-dir       "results/flavi_holdout_scores/${V}" \
        --device        cuda:0 \
        --tau           8.0 \
        --tag-prefix    "${V}_"
done

echo "=============================="
echo "Flavivirus NS3 pipeline complete."
echo "Stage 2 results: results/flavi_ablations/"
echo "Holdout scores:  results/flavi_holdout_scores/"
echo "=============================="
