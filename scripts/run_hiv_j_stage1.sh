#!/usr/bin/env bash
set -euo pipefail
set -x

if [[ $# -lt 1 ]]; then
  echo "usage: $0 J_VALUE" >&2
  echo "example: $0 25" >&2
  exit 2
fi

J_VALUE="$1"

cd /scratch/pranamlab/kimberly/PROPHET
export PYTHONPATH="/scratch/pranamlab/kimberly/PROPHET:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/scratch/pranamlab/kimberly/PROPHET/venv/bin/python}"
BASE_DIR="${BASE_DIR:-results/hiv_j_sweep}"
OUT_DIR="${BASE_DIR}/J_${J_VALUE}"
TREE="${TREE:-data/pre_stage1_split/trees/train/hiv_train_tree.nwk}"
FASTA="${FASTA:-data/pre_stage1_split/alignments/train/hiv_train_aligned.fasta}"
TREES_FILE="${TREES_FILE:-${OUT_DIR}/train_trees_paths.txt}"
PREFIX="${PREFIX:-hiv_train_J_${J_VALUE}}"
T_EVO="${T_EVO:-1.0}"
ENERGY_MODE="${ENERGY_MODE:-dca_plus_qi}"
SEED="${SEED:-42}"
TREE_SUBSAMPLE_SEED="${TREE_SUBSAMPLE_SEED:-42}"
SAMPLE_VARIANTS="${SAMPLE_VARIANTS:-500}"
BURN_IN="${BURN_IN:-200}"

mkdir -p "${OUT_DIR}"
find data/pre_stage1_split/trees/train -maxdepth 1 -type f -name '*_train_tree.nwk' | sort > "${TREES_FILE}"

echo "[start] J=${J_VALUE} trees_file=${TREES_FILE} t_evo=${T_EVO} $(date)"
trap 'status=$?; echo "[exit] J=${J_VALUE} status=${status} $(date)"; exit ${status}' EXIT

"${PYTHON}" prophet/stage1.py \
  --tree "${TREE}" \
  --trees-file "${TREES_FILE}" \
  --tree-subsample-j "${J_VALUE}" \
  --tree-subsample-seed "${TREE_SUBSAMPLE_SEED}" \
  --fasta "${FASTA}" \
  --prefix "${PREFIX}" \
  --out-dir "${OUT_DIR}" \
  --sample-variants "${SAMPLE_VARIANTS}" \
  --burn-in "${BURN_IN}" \
  --t-evo "${T_EVO}" \
  --energy-mode "${ENERGY_MODE}" \
  --seed "${SEED}" \
  --protein

echo "[done] J=${J_VALUE} $(date)"
