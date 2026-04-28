#!/usr/bin/env bash
set -euo pipefail

# Launch Stage 2 in nohup mode for each Stage 1 output prefix.
#
# Usage:
#   ./run_stage2_all_nohup.sh "<WT_SEQ>" /abs/path/to/dfm_ckpt.pt \
#     [STAGE1_DIR] [OUT_DIR] [LOG_DIR]
#
# Notes:
# - Stage 2 consumes variants + WT sequence (not lambda/Qi directly).
# - For each prefix, we try "${STAGE1_DIR}/${prefix}_gibbs_variants.fasta".
# - If that does not exist, we fall back to $VARIANTS_FASTA.
# - Jobs are launched in background with nohup and throttled by MAX_JOBS.

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"<WT_SEQ>\" <DFM_CKPT> [STAGE1_DIR] [OUT_DIR] [LOG_DIR]"
  exit 1
fi

WT_SEQ="$1"
DFM_CKPT="$2"
STAGE1_DIR="${3:-hadsbm-hiv/data/prophet/all_trees_stage1_protein}"
OUT_DIR="${4:-hadsbm-hiv/data/prophet/all_trees_stage2}"
LOG_DIR="${5:-hadsbm-hiv/data/prophet/all_trees_stage2_logs}"

# Fallback variants FASTA used when per-prefix variants are unavailable.
VARIANTS_FASTA="${VARIANTS_FASTA:-hadsbm-hiv/data/prophet/hiv_rerun_gibbs_variants.fasta}"
# Parallel job cap to avoid spawning too many simultaneous runs.
MAX_JOBS="${MAX_JOBS:-1}"
# Comma-separated GPU ids used in round-robin assignment.
GPU_IDS="${GPU_IDS:-1,2,3,4,5,7}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ ! -d "$STAGE1_DIR" ]]; then
  echo "Stage1 directory not found: $STAGE1_DIR"
  exit 1
fi

if [[ ! -f "$VARIANTS_FASTA" ]]; then
  echo "Fallback variants FASTA not found: $VARIANTS_FASTA"
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "No GPUs parsed from GPU_IDS=$GPU_IDS"
  exit 1
fi

shopt -s nullglob
lambda_files=( "$STAGE1_DIR"/*_lambda.npy )
shopt -u nullglob

if [[ ${#lambda_files[@]} -eq 0 ]]; then
  echo "No *_lambda.npy files found in: $STAGE1_DIR"
  exit 1
fi

echo "Launching Stage 2 nohup jobs:"
echo "  Stage1 dir  : $STAGE1_DIR"
echo "  Output dir  : $OUT_DIR"
echo "  Log dir     : $LOG_DIR"
echo "  Max jobs    : $MAX_JOBS"
echo "  GPU ids     : $GPU_IDS"
echo "  Fallback var: $VARIANTS_FASTA"
echo

launched=0
for lam in "${lambda_files[@]}"; do
  base="$(basename "$lam")"
  prefix="${base%_lambda.npy}"

  # Normalize known Stage-1 suffixes to keep output filenames cleaner.
  tree_prefix="$prefix"
  tree_prefix="${tree_prefix%_stage1p}"
  tree_prefix="${tree_prefix%_stage1}"

  per_tree_variants="$STAGE1_DIR/${prefix}_gibbs_variants.fasta"
  variants="$VARIANTS_FASTA"
  if [[ -f "$per_tree_variants" ]]; then
    variants="$per_tree_variants"
  fi

  out_json="$OUT_DIR/${tree_prefix}_stage2_designs.json"
  log_file="$LOG_DIR/${tree_prefix}.log"
  gpu_idx=$((launched % ${#GPU_LIST[@]}))
  gpu_id="${GPU_LIST[$gpu_idx]}"

  nohup env DEVICE="cuda:${gpu_id}" DFM_DEVICE="cuda:${gpu_id}" \
    bash "hadsbm-hiv/prophet/run_stage2.sh" \
    "$WT_SEQ" \
    "$DFM_CKPT" \
    "$variants" \
    "$out_json" \
    > "$log_file" 2>&1 &

  launched=$((launched + 1))
  echo "[$launched/${#lambda_files[@]}] launched ${tree_prefix} (pid=$!, gpu=${gpu_id}, variants=$(basename "$variants"))"

  # Throttle: keep at most MAX_JOBS live background jobs from this shell.
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
    sleep 1
  done
done

echo
echo "All jobs submitted: $launched"
echo "Use: ls \"$LOG_DIR\""
echo "Watch one log: tail -f \"$LOG_DIR/<tree_prefix>.log\""
