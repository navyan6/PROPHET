#!/usr/bin/env bash
set -u
cd /scratch/pranamlab/kimberly/PROPHET
RUN_DIR="results/hiv_stage2_raw_20260429_112329"
mkdir -p "${RUN_DIR}/logs"
echo "[launcher-start] $(date) run_dir=${RUN_DIR}"
status=0
OUT_DIR="${RUN_DIR}" scripts/run_hiv_stage2_mode.sh prophet 0 > "${RUN_DIR}/logs/prophet_gpu0.log" 2>&1 & p1=$!
OUT_DIR="${RUN_DIR}" scripts/run_hiv_stage2_mode.sh wt_only 4 > "${RUN_DIR}/logs/wt_only_gpu4.log" 2>&1 & p2=$!
OUT_DIR="${RUN_DIR}" scripts/run_hiv_stage2_mode.sh random_variants 5 > "${RUN_DIR}/logs/random_variants_gpu5.log" 2>&1 & p3=$!
OUT_DIR="${RUN_DIR}" scripts/run_hiv_stage2_mode.sh uniform_leaves 6 > "${RUN_DIR}/logs/uniform_leaves_gpu6.log" 2>&1 & p4=$!
for p in $p1 $p2 $p3 $p4; do
  wait "$p" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "[launcher-fail] Stage 2 batch failed $(date)"
  exit "$status"
fi
echo "[launcher] Stage 2 batch complete $(date)"
STAGE2_OUT="${PWD}/${RUN_DIR}" PEPTUNE_OUT_DIR="${PWD}/${RUN_DIR}/peptune_hiv" scripts/run_peptune_hiv_baseline.sh 0 500 10 > "${RUN_DIR}/logs/peptune_gpu0.log" 2>&1 & b1=$!
STAGE2_OUT="${PWD}/${RUN_DIR}" PEPTUNE_OUT_DIR="${PWD}/${RUN_DIR}/peptune_hiv" scripts/run_peptune_unconditional_hiv_baseline.sh 4 500 10 > "${RUN_DIR}/logs/peptune_unconditional_gpu4.log" 2>&1 & b2=$!
STAGE2_OUT="${PWD}/${RUN_DIR}" RFDIFFUSION_OUT_DIR="${PWD}/${RUN_DIR}/rfdiffusion_hiv" scripts/run_rfdiffusion_hiv_baseline.sh 5 500 10 > "${RUN_DIR}/logs/rfdiffusion_gpu5.log" 2>&1 & b3=$!
for p in $b1 $b2 $b3; do
  wait "$p" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "[launcher-fail] Baseline batch failed $(date)"
  exit "$status"
fi
venv/bin/python scripts/build_hiv_paper_tables.py --stage2-dir "${RUN_DIR}" --out-dir "${RUN_DIR}/tables" > "${RUN_DIR}/logs/build_tables.log" 2>&1 || status=1
echo "[launcher-done] status=${status} $(date)"
exit "${status}"
