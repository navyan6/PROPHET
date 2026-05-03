#!/usr/bin/env bash
set -euo pipefail
set -x

if [[ $# -lt 1 ]]; then
  echo "usage: $0 M_VALUE" >&2
  echo "example: $0 500" >&2
  exit 2
fi

M_VALUE="$1"

cd /scratch/pranamlab/kimberly/PROPHET

PYTHON="${PYTHON:-/scratch/pranamlab/kimberly/PROPHET/venv/bin/python}"
export PYTHONPATH="/scratch/pranamlab/kimberly/PROPHET${PYTHONPATH:+:${PYTHONPATH}}"
BASE_DIR="${BASE_DIR:-results/hiv_m_sweep}"
OUT_DIR="${BASE_DIR}/M_${M_VALUE}"
TREE="${TREE:-data/pre_stage1_split/trees/train/hiv_train_tree.nwk}"
FASTA="${FASTA:-data/pre_stage1_split/alignments/train/hiv_train_aligned.fasta}"
PREFIX="${PREFIX:-hiv_train_M_${M_VALUE}}"
T_EVO="${T_EVO:-1.0}"
ENERGY_MODE="${ENERGY_MODE:-dca_plus_qi}"
SEED="${SEED:-42}"
BURN_IN="${BURN_IN:-200}"

mkdir -p "${OUT_DIR}"

echo "[start] M=${M_VALUE} t_evo=${T_EVO} $(date)"
trap 'status=$?; echo "[exit] M=${M_VALUE} status=${status} $(date)"; exit ${status}' EXIT

"${PYTHON}" prophet/stage1.py \
  --tree "${TREE}" \
  --fasta "${FASTA}" \
  --prefix "${PREFIX}" \
  --out-dir "${OUT_DIR}" \
  --sample-variants "${M_VALUE}" \
  --burn-in "${BURN_IN}" \
  --t-evo "${T_EVO}" \
  --energy-mode "${ENERGY_MODE}" \
  --seed "${SEED}" \
  --protein

echo "[done] M=${M_VALUE} $(date)"
