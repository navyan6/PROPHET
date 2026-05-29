#!/bin/bash
# Spike Stage 2: all Table 2 + Table 4 ablations in parallel across 8 GPUs,
# followed by holdout robustness scoring.
# Prerequisite: run_spike_stage1_all.slurm must have completed.
#
#SBATCH --job-name=spike_s2_all
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:B200:8
#SBATCH --mem=384G
#SBATCH --time=36:00:00
#SBATCH --output=logs/spike_s2_all_%j.out
#SBATCH --error=logs/spike_s2_all_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/nnori/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
export PATH=/vast/home/n/nnori/.conda/envs/hadsbm/bin:$PATH
export HF_HOME=/vast/projects/pranam/lab/nnori/.cache/huggingface

set -e
cd "$REPO"
mkdir -p logs results/spike_ablations results/spike_holdout_scores

STAMP=$(date +%Y%m%d_%H%M%S)
HOLDOUT=data/spike/alignments/test/spike_test_clade_holdout.fasta
UNFILTERED=results/spike_prophet_final/spike_prophet_gibbs_variants.fasta
STAGE2=prophet/stage2.py

# ESM-filtered variants (primary); fall back to unfiltered if not found
VARIANTS=$(ls results/spike_prophet_final/spike_prophet_*esm_filtered*.fasta 2>/dev/null | head -1)
[ -z "$VARIANTS" ] && VARIANTS=$UNFILTERED

# Count variants for --variant-limit (use all available, capped at 500)
N_VARIANTS=$(grep -c "^>" "$VARIANTS" 2>/dev/null || echo 500)
VARIANT_LIMIT=$(( N_VARIANTS < 500 ? N_VARIANTS : 500 ))
echo "Using $N_VARIANTS variants from $VARIANTS (limit=$VARIANT_LIMIT)"

# Consensus WT from training alignment
WT_SEQ=$($PYTHON -c "
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
recs = list(SeqIO.parse('data/spike/alignments/train/spike_train_aligned.fasta', 'fasta'))
aln = MultipleSeqAlignment(recs)
wt = ''
for i in range(aln.get_alignment_length()):
    col = aln[:, i]
    counts = {}
    for aa in col:
        if aa != '-': counts[aa] = counts.get(aa, 0) + 1
    wt += max(counts, key=counts.get) if counts else 'X'
print(wt)
")
echo "WT consensus length: ${#WT_SEQ}"

# Common stage2 args
S2_COMMON="
    --wt-seq                  $WT_SEQ
    --device                  cuda:0
    --dfm-device              cuda:0
    --n-designs               500
    --n-steps                 200
    --peptide-length          10
    --beta                    5.0
    --eta                     0.1
    --tau-bind                8.0
    --peptiverse-normalization raw
    --variant-limit           $VARIANT_LIMIT
    --guidance-var-limit      50
    --verbose-sampling
"

# Skips if JSON for this tag already exists
needs_run() {
    compgen -G "results/spike_ablations/${1}-*.json" > /dev/null 2>&1 && return 1 || return 0
}

run_exp() {
    local name=$1 fasta=$2 gpu=$3 extra=$4
    if ! needs_run "$name"; then
        echo "[skip] $name already has results"
        return
    fi
    if [ ! -f "$fasta" ]; then
        echo "[skip] $name: variant FASTA not found: $fasta"
        return
    fi
    echo "Launching $name on GPU $gpu ..."
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
    $PYTHON $STAGE2 $S2_COMMON \
        --variants-fasta "$fasta" \
        --out-json       "results/spike_ablations/${name}-${STAMP}.json" \
        $extra \
        > "results/spike_ablations/${name}-${STAMP}.log" 2>&1 &
}

echo "=============================="
echo "Job $SLURM_JOB_ID — Spike Stage 2 (all ablations)"
echo "=============================="

# ── Table 2: method comparison (GPUs 0–4) ──────────────────────────────────
run_exp "t2_prophet"           "$VARIANTS"    0 ""
run_exp "t2_wt_only"           "$VARIANTS"    1 "--design-mode wt_only"
run_exp "t2_random_variants"   "$VARIANTS"    2 "--design-mode random_variants"
run_exp "t2_uniform_leaves"    "$VARIANTS"    3 "--design-mode uniform_leaves --guidance-variants-fasta data/spike/alignments/train/spike_train_aligned.fasta"
run_exp "t2_esm_only_variants" "$VARIANTS"    4 "--design-mode esm_only_variants"

# ── Table 4: Stage-1 ablations (GPUs 5–6) ──────────────────────────────────
run_exp "t4_no_dca"    "results/spike_stage1_ablations/spike_no_dca_gibbs_variants.fasta"    5 "--variant-limit $VARIANT_LIMIT"
run_exp "t4_no_lambda" "results/spike_stage1_ablations/spike_no_lambda_gibbs_variants.fasta" 6 "--variant-limit $VARIANT_LIMIT"
run_exp "t4_no_esm"    "$UNFILTERED"                                                         7 "--variant-limit $VARIANT_LIMIT"

echo "Waiting for batch 1 (Table 2 + Table 4 Stage 1 ablations) ..."
wait

# ── Table 4: CVaR η sensitivity (GPUs 0–2, reuse after batch 1) ───────────
run_exp "t5_eta_0.1" "$VARIANTS" 0 "--eta 0.1"
run_exp "t5_eta_0.5" "$VARIANTS" 1 "--eta 0.5"
run_exp "t5_eta_1.0" "$VARIANTS" 2 "--eta 1.0"

echo "Waiting for batch 2 (CVaR eta sweep) ..."
wait

echo "All Stage 2 experiments complete."

# ── Holdout robustness scoring ──────────────────────────────────────────────
if [ ! -f "$HOLDOUT" ]; then
    echo "ERROR: holdout FASTA not found: $HOLDOUT"
    exit 1
fi

echo "Running holdout robustness scoring ..."
$PYTHON scripts/score_holdout_robustness.py \
    --ablations-dir results/spike_ablations \
    --holdout-fasta "$HOLDOUT" \
    --out-dir       results/spike_holdout_scores \
    --device        cuda:0 \
    --tau           8.0

echo "=============================="
echo "Spike pipeline complete."
echo "Stage 2 results:  results/spike_ablations/"
echo "Holdout scores:   results/spike_holdout_scores/"
echo "=============================="
