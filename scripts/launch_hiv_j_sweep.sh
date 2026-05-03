#!/usr/bin/env bash
set -euo pipefail

cd /scratch/pranamlab/kimberly/PROPHET

BASE_DIR="${BASE_DIR:-results/hiv_j_sweep}"
J_VALUES="${J_VALUES:-1 5 10 25 50 100 144}"

mkdir -p "${BASE_DIR}/logs"

read -r -a J_ARRAY <<< "${J_VALUES}"

echo "[launcher start] $(date)"
echo "[launcher config] J_VALUES=${J_VALUES} BASE_DIR=${BASE_DIR}"

for J in "${J_ARRAY[@]}"; do
  log="${BASE_DIR}/J_${J}/hiv_train_J_${J}_stage1.log"
  mkdir -p "${BASE_DIR}/J_${J}"
  echo "[stage1] J=${J} log=${log}"
  BASE_DIR="${BASE_DIR}" scripts/run_hiv_j_stage1.sh "${J}" > "${log}" 2>&1
done

echo "[launcher done] $(date)"
