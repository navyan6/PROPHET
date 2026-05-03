#!/usr/bin/env bash
set -euo pipefail

cd /scratch/pranamlab/kimberly/PROPHET

BASE_DIR="${BASE_DIR:-results/hiv_m_sweep}"
M_VALUES="${M_VALUES:-50 100 250 500}"
GPUS="${GPUS:-1 2 3 4}"

mkdir -p "${BASE_DIR}/logs"

read -r -a M_ARRAY <<< "${M_VALUES}"
read -r -a GPU_ARRAY <<< "${GPUS}"

if [[ "${#GPU_ARRAY[@]}" -lt "${#M_ARRAY[@]}" ]]; then
  echo "need at least as many GPUS as M_VALUES" >&2
  echo "M_VALUES=${M_VALUES}" >&2
  echo "GPUS=${GPUS}" >&2
  exit 2
fi

echo "[launcher start] $(date)"
echo "[launcher config] M_VALUES=${M_VALUES} GPUS=${GPUS} BASE_DIR=${BASE_DIR}"

for idx in "${!M_ARRAY[@]}"; do
  M="${M_ARRAY[$idx]}"
  log="${BASE_DIR}/M_${M}/hiv_train_M_${M}_stage1.log"
  mkdir -p "${BASE_DIR}/M_${M}"
  echo "[stage1] M=${M} log=${log}"
  BASE_DIR="${BASE_DIR}" scripts/run_hiv_m_stage1.sh "${M}" > "${log}" 2>&1
done

for idx in "${!M_ARRAY[@]}"; do
  M="${M_ARRAY[$idx]}"
  gpu="${GPU_ARRAY[$idx]}"
  log="${BASE_DIR}/M_${M}/hiv_train_M_${M}_stage2_gpu${gpu}.log"
  echo "[launch stage2] M=${M} gpu=${gpu} log=${log}"
  BASE_DIR="${BASE_DIR}" nohup scripts/run_hiv_m_stage2.sh "${M}" "${gpu}" > "${log}" 2>&1 &
  echo "[pid] M=${M} pid=$!"
done

wait
echo "[launcher done] $(date)"
