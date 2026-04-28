#!/usr/bin/env bash
set -euo pipefail

# General Stage 2 launcher.
# Usage:
#   ./run_stage2.sh "<WT_SEQ>" /abs/path/to/dfm_ckpt.pt [variants_fasta] [out_json]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"<WT_SEQ>\" <DFM_CKPT> [VARIANTS_FASTA] [OUT_JSON]"
  exit 1
fi

WT_SEQ="$1"
DFM_CKPT="$2"
VARIANTS_FASTA="${3:-hadsbm-hiv/data/prophet/hiv_rerun_gibbs_variants.fasta}"
OUT_JSON="${4:-hadsbm-hiv/data/prophet/stage2_designs.json}"

# Optional tuning via env vars with sensible defaults.
N_DESIGNS="${N_DESIGNS:-500}"
N_STEPS="${N_STEPS:-200}"
PEPTIDE_LENGTH="${PEPTIDE_LENGTH:-10}"
ETA="${ETA:-0.1}"
BETA="${BETA:-5.0}"
DELTA_ALPHA="${DELTA_ALPHA:-1.0}"
HYPERCONE_ANGLE="${HYPERCONE_ANGLE:-45.0}"
VARIANT_LIMIT="${VARIANT_LIMIT:-}"
DEVICE="${DEVICE:-cpu}"
DFM_DEVICE="${DFM_DEVICE:-$DEVICE}"
AFFINITY_MODE="${AFFINITY_MODE:-surrogate}"
SEED="${SEED:-42}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CMD=(
  python hadsbm-hiv/prophet/stage2.py
  --variants-fasta "$VARIANTS_FASTA"
  --wt-seq "$WT_SEQ"
  --dfm-ckpt "$DFM_CKPT"
  --out-json "$OUT_JSON"
  --n-designs "$N_DESIGNS"
  --n-steps "$N_STEPS"
  --peptide-length "$PEPTIDE_LENGTH"
  --eta "$ETA"
  --beta "$BETA"
  --delta-alpha "$DELTA_ALPHA"
  --hypercone-angle "$HYPERCONE_ANGLE"
  --affinity-mode "$AFFINITY_MODE"
  --device "$DEVICE"
  --dfm-device "$DFM_DEVICE"
  --seed "$SEED"
)

if [[ -n "$VARIANT_LIMIT" ]]; then
  CMD+=(--variant-limit "$VARIANT_LIMIT")
fi

echo "Running Stage 2 with:"
echo "  WT length: ${#WT_SEQ}"
echo "  Checkpoint: $DFM_CKPT"
echo "  Variants: $VARIANTS_FASTA"
echo "  Output: $OUT_JSON"
echo

"${CMD[@]}"
