#!/usr/bin/env bash
set -euo pipefail

# Quick smoke test run (small/fast settings).
# Usage:
#   ./run_stage2_quick.sh "<WT_SEQ>" /abs/path/to/dfm_ckpt.pt

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"<WT_SEQ>\" <DFM_CKPT>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_DESIGNS="${N_DESIGNS:-40}" \
N_STEPS="${N_STEPS:-60}" \
VARIANT_LIMIT="${VARIANT_LIMIT:-120}" \
DEVICE="${DEVICE:-cpu}" \
DFM_DEVICE="${DFM_DEVICE:-$DEVICE}" \
"$SCRIPT_DIR/run_stage2.sh" "$1" "$2" \
  "hadsbm-hiv/data/prophet/hiv_rerun_gibbs_variants.fasta" \
  "hadsbm-hiv/data/prophet/stage2_designs_quick.json"
