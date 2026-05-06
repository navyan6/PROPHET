#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU_LIST [OUT_BASE]" >&2
  echo "example: $0 0,1,2,3 results/hiv_stage2_m_sweep" >&2
  echo "override: M_VALUES='50 100 250 500' $0 0,1,2,3" >&2
  exit 2
fi

GPU_LIST="$1"
OUT_BASE="${2:-${OUT_BASE:-results/hiv_stage2_m_sweep}}"

cd /scratch/pranamlab/kimberly/PROPHET

exec scripts/run_hiv_M.sh "${GPU_LIST}" "${OUT_BASE}"
