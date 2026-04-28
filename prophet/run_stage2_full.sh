#!/usr/bin/env bash
set -euo pipefail

# Full/production Stage 2 run defaults.
# Usage:
#   ./run_stage2_full.sh "<WT_SEQ>" /abs/path/to/dfm_ckpt.pt

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"<WT_SEQ>\" <DFM_CKPT>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_DESIGNS="${N_DESIGNS:-1200}" \
N_STEPS="${N_STEPS:-300}" \
PEPTIDE_LENGTH="${PEPTIDE_LENGTH:-10}" \
ETA="${ETA:-0.1}" \
BETA="${BETA:-5.0}" \
DELTA_ALPHA="${DELTA_ALPHA:-1.0}" \
HYPERCONE_ANGLE="${HYPERCONE_ANGLE:-45.0}" \
VARIANT_LIMIT="${VARIANT_LIMIT:-}" \
DEVICE="${DEVICE:-cpu}" \
DFM_DEVICE="${DFM_DEVICE:-$DEVICE}" \
AFFINITY_MODE="${AFFINITY_MODE:-surrogate}" \
SEED="${SEED:-42}" \
"$SCRIPT_DIR/run_stage2.sh" "$1" "$2" \
  "hadsbm-hiv/data/prophet/hiv_rerun_gibbs_variants.fasta" \
  "hadsbm-hiv/data/prophet/stage2_designs_full.json"
