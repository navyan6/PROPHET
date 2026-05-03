#!/usr/bin/env bash
set -euo pipefail

cd /scratch/pranamlab/kimberly/PROPHET

declare -A GPUS=(
  ["0.5"]=1
  ["1.0"]=2
  ["2.0"]=3
  ["5.0"]=4
)

echo "[launcher start] $(date)"

for tevo in 0.5 1.0 2.0 5.0; do
  gpu="${GPUS[$tevo]}"
  log="results/hiv_tevo_sweep/tevo_${tevo}/hiv_train_tevo_${tevo}_stage2_gpu${gpu}.log"
  echo "[launch] tevo=${tevo} gpu=${gpu} log=${log}"
  nohup scripts/run_hiv_tevo_stage2.sh "${tevo}" "${gpu}" > "${log}" 2>&1 &
  echo "[pid] tevo=${tevo} pid=$!"
done

wait
echo "[launcher done] $(date)"
