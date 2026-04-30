#!/usr/bin/env bash
set -euo pipefail

cd /scratch/pranamlab/kimberly/PROPHET

PYTHON=/scratch/pranamlab/kimberly/PROPHET/venv/bin/python
OUT_DIR="${OUT_DIR:-results/hiv_stage2}"

required=(
  "${OUT_DIR}/hiv_train_prophet_robust_design.json"
  "${OUT_DIR}/hiv_train_wt_only_robust_design.json"
  "${OUT_DIR}/hiv_train_random_variants_robust_design.json"
  "${OUT_DIR}/hiv_train_uniform_leaves_robust_design.json"
)

echo "[table-watch-start] $(date)"
while true; do
  missing=0
  for path in "${required[@]}"; do
    if [[ ! -s "${path}" ]]; then
      missing=1
      break
    fi
  done
  if [[ "${missing}" -eq 0 ]]; then
    break
  fi
  sleep 120
done

"${PYTHON}" scripts/build_hiv_paper_tables.py \
  --stage2-dir "${OUT_DIR}" \
  --out-dir "${OUT_DIR}/tables"

echo "[table-watch-done] $(date)"
