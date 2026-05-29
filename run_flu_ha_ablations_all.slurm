#!/bin/bash
# Flu HA: Stage 1 ablations (-DCA, -λ), then all Stage 2 experiments in parallel,
# then holdout robustness scoring.
# Prerequisite: prophet_flu_ha (Stage 1 main run) must have completed.
# Submit: sbatch --dependency=afterok:<flu_ha_stage1_job_id> run_flu_ha_ablations_all.slurm
#
#SBATCH --job-name=flu_ha_abl
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:B200:8
#SBATCH --mem=384G
#SBATCH --time=36:00:00
#SBATCH --output=logs/flu_ha_abl_%j.out
#SBATCH --error=logs/flu_ha_abl_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/nnori/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
export PATH=/vast/home/n/nnori/.conda/envs/hadsbm/bin:$PATH
export HF_HOME=/vast/projects/pranam/lab/nnori/.cache/huggingface

set -e
cd "$REPO"
mkdir -p logs results/flu_ha_ablations results/flu_ha_stage1_ablations results/flu_ha_holdout_scores

STAMP=$(date +%Y%m%d_%H%M%S)

TREE=data/flu_ha/trees/train/flu_ha_train_tree.nwk
TREES_FILE=data/flu_ha/trees/train/flu_ha_bootstrap_trees.txt
FASTA=data/flu_ha/alignments/train/flu_ha_train_aligned.fasta
HOLDOUT=data/flu_ha/alignments/test/flu_ha_test_clade_holdout.fasta
UNFILTERED=results/flu_ha_prophet_final/flu_ha_prophet_gibbs_variants.fasta
STAGE1=prophet/stage1.py
STAGE2=prophet/stage2.py

# ESM-filtered variants (primary); fall back to unfiltered if not found
VARIANTS=$(ls results/flu_ha_prophet_final/flu_ha_prophet_*esm_filtered*.fasta 2>/dev/null | head -1)
[ -z "$VARIANTS" ] && VARIANTS=$UNFILTERED

echo "=============================="
echo "Job $SLURM_JOB_ID — Flu HA ablations + Stage 2 + holdout"
echo "=============================="
echo "Variants FASTA: $VARIANTS"

# Count variants for --variant-limit (capped at 500)
N_VARIANTS=$(grep -c "^>" "$VARIANTS" 2>/dev/null || echo 500)
VARIANT_LIMIT=$(( N_VARIANTS < 500 ? N_VARIANTS : 500 ))
echo "Using $N_VARIANTS variants (limit=$VARIANT_LIMIT)"

# Consensus WT from training alignment
WT_SEQ=$($PYTHON -c "
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
recs = list(SeqIO.parse('$FASTA', 'fasta'))
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

# Shared Stage 1 args (used for ablations only — main run already done)
S1_ABL_ARGS="
    --tree          $TREE
    --trees-file    $TREES_FILE
    --fasta         $FASTA
    --protein
    --n-bootstraps  1
    --sample-variants 500
    --burn-in       200
    --t-evo         0.15
    --energy-mode   paper_dca
    --ensemble-mode
    --esm-device    cuda:0
    --esm-model     facebook/esm2_t33_650M_UR50D
    --esm-filter-delta 20
    --conserv-weight 0.05
    --seed          42
"

# ── Phase 1: Stage 1 ablations (parallel on up to 4 GPUs) ──────────────────
echo "--- Phase 1: Stage 1 ablations ---"

run_s1_abl() {
    local abl=$1 gpu=$2
    local out_fasta="results/flu_ha_stage1_ablations/flu_ha_${abl}_gibbs_variants.fasta"
    local n
    n=$(grep -c '^>' "$out_fasta" 2>/dev/null || true); n=${n:-0}
    if [ "$n" -gt 0 ]; then
        echo "[skip] $abl: already exists ($n variants)"; return
    fi
    echo "Launching Stage 1 ablation: $abl on GPU $gpu ..."
    local extra="" conserv_w="0.05" ensemble="--ensemble-mode --trees-file $TREES_FILE"
    [ "$abl" = "no_dca" ]       && extra="--ablate-zero-dca-couplings"
    [ "$abl" = "no_lambda" ]    && extra="--ablate-flatten-lambda"
    [ "$abl" = "no_conserv" ]   && conserv_w="0.0"
    [ "$abl" = "no_bootstrap" ] && ensemble=""
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PYTHON $STAGE1 \
        --tree           "$TREE" \
        --fasta          "$FASTA" \
        --protein \
        --out-dir        results/flu_ha_stage1_ablations \
        --prefix         "flu_ha_${abl}" \
        --n-bootstraps   1 \
        --sample-variants 500 \
        --burn-in        200 \
        --t-evo          0.15 \
        --energy-mode    paper_dca \
        $ensemble \
        --esm-device     cuda:0 \
        --esm-model      facebook/esm2_t33_650M_UR50D \
        --esm-filter-delta 20 \
        --conserv-weight "$conserv_w" \
        --seed           42 \
        $extra \
        > "logs/flu_ha_s1_abl_${abl}_${SLURM_JOB_ID}.log" 2>&1 &
}

gpu=0
for ABL in no_dca no_lambda no_conserv no_bootstrap; do
    run_s1_abl "$ABL" $gpu
    gpu=$(( gpu + 1 ))
done
wait; echo "Stage 1 ablations complete."

# ── Common Stage 2 args ─────────────────────────────────────────────────────
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
    --seed                    42
"

# Skips if JSON for this tag already exists
needs_run() {
    compgen -G "results/flu_ha_ablations/${1}-*.json" > /dev/null 2>&1 && return 1 || return 0
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
        --out-json       "results/flu_ha_ablations/${name}-${STAMP}.json" \
        $extra \
        > "results/flu_ha_ablations/${name}-${STAMP}.log" 2>&1 &
}

# ── Phase 2 Batch 1: Table 2 + Table 4 Stage 1 ablations (8 GPUs) ──────────
echo "--- Phase 2 Batch 1: Table 2 + Table 4 ---"

run_exp "t2_prophet"           "$VARIANTS"    0 ""
run_exp "t2_wt_only"           "$VARIANTS"    1 "--design-mode wt_only"
run_exp "t2_random_variants"   "$VARIANTS"    2 "--design-mode random_variants"
run_exp "t2_uniform_leaves"    "$VARIANTS"    3 "--design-mode uniform_leaves --guidance-variants-fasta data/flu_ha/alignments/train/flu_ha_train_aligned.fasta"
run_exp "t2_esm_only_variants" "$VARIANTS"    4 "--design-mode esm_only_variants"
run_exp "t4_no_dca"    "results/flu_ha_stage1_ablations/flu_ha_no_dca_gibbs_variants.fasta"    5 "--variant-limit $VARIANT_LIMIT"
run_exp "t4_no_lambda" "results/flu_ha_stage1_ablations/flu_ha_no_lambda_gibbs_variants.fasta" 6 "--variant-limit $VARIANT_LIMIT"
run_exp "t4_no_esm"    "$UNFILTERED"                                                           7 "--variant-limit $VARIANT_LIMIT"

echo "Waiting for batch 1 (Table 2 + Table 4 Stage 1 ablations) ..."
wait

# ── Phase 2 Batch 1b: -Conservation / -Bootstrap Stage 2 ───────────────────
echo "--- Phase 2 Batch 1b: t4_no_conserv / t4_no_bootstrap ---"
run_exp "t4_no_conserv" \
    "results/flu_ha_stage1_ablations/flu_ha_no_conserv_gibbs_variants.fasta" \
    0 "--variant-limit $VARIANT_LIMIT"
run_exp "t4_no_bootstrap" \
    "results/flu_ha_stage1_ablations/flu_ha_no_bootstrap_gibbs_variants.fasta" \
    1 "--variant-limit $VARIANT_LIMIT"
echo "Waiting for batch 1b ..."
wait

# ── Phase 2 Batch 2: CVaR η sensitivity (3 GPUs) ───────────────────────────
echo "--- Phase 2 Batch 2: CVaR eta sweep ---"

run_exp "t5_eta_0.1" "$VARIANTS" 0 "--eta 0.1"
run_exp "t5_eta_0.5" "$VARIANTS" 1 "--eta 0.5"
run_exp "t5_eta_1.0" "$VARIANTS" 2 "--eta 1.0"

echo "Waiting for batch 2 (CVaR eta sweep) ..."
wait

# ── Phase 2 Batch 3: M sensitivity sweep ────────────────────────────────────
echo "--- Phase 2 Batch 3: M sweep (variant-limit sensitivity) ---"
gpu=0
for M in 50 100 250 500; do
    if [ "$M" -le "$VARIANT_LIMIT" ]; then
        MFASTA="$VARIANTS"
    else
        MFASTA="$UNFILTERED"
    fi
    run_exp "t6_M_${M}" "$MFASTA" $gpu "--variant-limit $M"
    gpu=$(( gpu + 1 ))
done
echo "Waiting for batch 3 (M sweep) ..."
wait

echo "All Stage 2 experiments complete."

# ── Phase 3: Holdout robustness scoring ────────────────────────────────────
echo "--- Phase 3: Holdout robustness scoring ---"

if [ ! -f "$HOLDOUT" ]; then
    echo "ERROR: holdout FASTA not found: $HOLDOUT"
    exit 1
fi

$PYTHON scripts/score_holdout_robustness.py \
    --ablations-dir results/flu_ha_ablations \
    --holdout-fasta "$HOLDOUT" \
    --out-dir       results/flu_ha_holdout_scores \
    --device        cuda:0 \
    --tau           8.0

echo "=============================="
echo "Flu HA pipeline complete."
echo "Stage 2 results:  results/flu_ha_ablations/"
echo "Holdout scores:   results/flu_ha_holdout_scores/"
echo "=============================="
