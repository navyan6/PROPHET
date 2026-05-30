#!/bin/bash
# Universal PROPHET pipeline script.
#
# Runs Stage 1 (DCA + Gibbs + ESM filter) then Stage 2 (Table 2 comparison,
# 5 modes in parallel) for any registered target.
#
# Usage:
#   sbatch --export=TARGET=hiv_protease run_prophet.slurm
#   sbatch --export=TARGET=hcv_ns3 run_prophet.slurm
#   sbatch --export=TARGET=flu_ha run_prophet.slurm
#
# Or use submit.sh for convenience:
#   ./submit.sh hiv_protease
#
# Registered targets: hiv_protease, hcv_ns3, flu_ha, flu_na,
#                     sars_mpro, zika_ns3, wnv_ns3
#
#SBATCH --job-name=prophet
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:B200:6
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prophet_%x_%j.out
#SBATCH --error=logs/prophet_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

set -e

REPO=/vast/projects/pranam/lab/nnori/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
export PATH=/vast/home/n/nnori/.conda/envs/hadsbm/bin:$PATH
export HF_HOME=/vast/projects/pranam/lab/nnori/.cache/huggingface
export TORCH_HOME=/vast/projects/pranam/lab/nnori/.cache/torch
export XDG_CACHE_HOME=/vast/projects/pranam/lab/nnori/.cache
export PYTHONUNBUFFERED=1

cd "$REPO"

# ------------------------------------------------------------------
# Validate TARGET
# ------------------------------------------------------------------
if [ -z "${TARGET:-}" ]; then
    echo "ERROR: TARGET is not set. Submit with --export=TARGET=<name>"
    echo "  Valid targets: hiv_protease hcv_ns3 flu_ha flu_na sars_mpro zika_ns3 wnv_ns3"
    exit 1
fi

mkdir -p logs

echo "=============================="
echo "PROPHET pipeline: $TARGET"
echo "Job $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=============================="

# ------------------------------------------------------------------
# Load target config from configs/targets.py
# ------------------------------------------------------------------
eval "$($PYTHON -c "
import sys, json
sys.path.insert(0, '.')
from configs.targets import get_target
cfg = get_target('$TARGET')
keys = ('alignment','tree','trees_file','holdout','out_prefix',
        't_evo','n_bootstraps','sample_variants','burn_in',
        'energy_mode','adaptive_l2','l2_reg_base',
        'esm_filter_delta_per_residue','esm_model',
        'conserv_weight','seed','protein','ensemble_mode')
for k, v in cfg.items():
    if k in keys:
        print('CFG_' + k.upper() + '=' + (json.dumps(str(v)) if v is not None else '""'))
")"

# ------------------------------------------------------------------
# Derived paths
# ------------------------------------------------------------------
STAGE1_OUT="results/$TARGET/stage1"
STAGE2_OUT="results/$TARGET/comparison"
PREFIX="$CFG_OUT_PREFIX"

mkdir -p "$STAGE1_OUT" "$STAGE2_OUT"

TREE="$CFG_TREE"
TREES_FILE="$CFG_TREES_FILE"
FASTA="$CFG_ALIGNMENT"
HOLDOUT="$CFG_HOLDOUT"

for F in "$TREE" "$TREES_FILE" "$FASTA"; do
    [ -f "$F" ] || { echo "ERROR: missing $F"; exit 1; }
done
echo "Train sequences: $(grep -c '^>' "$FASTA")"

# ------------------------------------------------------------------
# Stage 1 flags
# ------------------------------------------------------------------
ENSEMBLE_FLAG=""
[ "$CFG_ENSEMBLE_MODE" = "True" ] && ENSEMBLE_FLAG="--ensemble-mode"

ADAPTIVE_L2_FLAG=""
[ "$CFG_ADAPTIVE_L2" = "True" ] && ADAPTIVE_L2_FLAG="--adaptive-l2"

# ------------------------------------------------------------------
# Stage 1: DCA + Gibbs + ESM filter
# ------------------------------------------------------------------
echo ""
echo "--- Stage 1: DCA + Gibbs sampling ---"

$PYTHON prophet/stage1.py \
    --tree              "$TREE" \
    --trees-file        "$TREES_FILE" \
    --fasta             "$FASTA" \
    --out-dir           "$STAGE1_OUT" \
    --prefix            "$PREFIX" \
    --n-bootstraps      "$CFG_N_BOOTSTRAPS" \
    --sample-variants   "$CFG_SAMPLE_VARIANTS" \
    --burn-in           "$CFG_BURN_IN" \
    --t-evo             "$CFG_T_EVO" \
    --energy-mode       "$CFG_ENERGY_MODE" \
    $ENSEMBLE_FLAG \
    --esm-device        cuda:0 \
    --esm-model         "$CFG_ESM_MODEL" \
    ${CFG_ESM_FILTER_DELTA_PER_RESIDUE:+--esm-filter-delta-per-residue "$CFG_ESM_FILTER_DELTA_PER_RESIDUE"} \
    $ADAPTIVE_L2_FLAG \
    --l2-reg-base       "$CFG_L2_REG_BASE" \
    --conserv-weight    "$CFG_CONSERV_WEIGHT" \
    --seed              "$CFG_SEED"

echo ""
echo "Stage 1 complete."
N_GIBBS=$(grep -c "^>" "$STAGE1_OUT/${PREFIX}_gibbs_variants.fasta" 2>/dev/null || echo 0)
echo "  Gibbs variants: $N_GIBBS"
ESM_FASTA=$(ls "$STAGE1_OUT/${PREFIX}_"*esm_filtered*.fasta 2>/dev/null | head -1 || true)
if [ -n "$ESM_FASTA" ]; then
    N_ESM=$(grep -c "^>" "$ESM_FASTA" 2>/dev/null || echo 0)
    echo "  ESM-filtered variants: $N_ESM  ($ESM_FASTA)"
fi

if [ "${N_GIBBS:-0}" -eq 0 ]; then
    echo "ERROR: Stage 1 produced 0 Gibbs variants. Aborting Stage 2."
    exit 1
fi

# ------------------------------------------------------------------
# Stage 2: Table 2 comparison (5 modes, GPUs 1–5)
# ------------------------------------------------------------------
echo ""
echo "--- Stage 2: Table 2 comparison (5 modes) ---"

$PYTHON prophet/run_comparison.py \
    --target    "$TARGET" \
    --out-dir   "$STAGE2_OUT" \
    --gpus      1,2,3,4,5 \
    --seed      "$CFG_SEED"

echo ""
echo "=============================="
echo "Pipeline complete: $TARGET"
echo "  Stage 1: $STAGE1_OUT"
echo "  Stage 2: $STAGE2_OUT"
echo "Finished: $(date)"
echo "=============================="
