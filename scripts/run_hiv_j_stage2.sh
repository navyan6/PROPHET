#!/usr/bin/env bash
set -euo pipefail
set -x

if [[ $# -lt 2 ]]; then
  echo "usage: $0 J_VALUE GPU_INDEX" >&2
  echo "example: $0 25 8" >&2
  exit 2
fi

J_VALUE="$1"
GPU_INDEX="$2"

cd /scratch/pranamlab/kimberly/PROPHET

PYTHON="${PYTHON:-/scratch/pranamlab/kimberly/PROPHET/venv/bin/python}"
export PYTHONPATH="/scratch/pranamlab/kimberly/PROPHET${PYTHONPATH:+:${PYTHONPATH}}"
BASE_DIR="${BASE_DIR:-results/hiv_j_sweep}"
OUT_DIR="${BASE_DIR}/J_${J_VALUE}"
PREFIX="${PREFIX:-hiv_train_J_${J_VALUE}}"
VARIANTS_FASTA="${OUT_DIR}/${PREFIX}_gibbs_variants.fasta"
ESCAPE_FASTA=data/pre_stage1_split/alignments/test/hiv_test_aligned.fasta
WT_SEQ=PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF
DFM_CKPT=MOG-DFM/ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt

mkdir -p "${OUT_DIR}"

DESIGNS_JSON="${OUT_DIR}/${PREFIX}_stage2_peptiverse.json"
PARETO_JSON="${OUT_DIR}/${PREFIX}_pareto.json"
ETA_JSON="${OUT_DIR}/${PREFIX}_eta_sensitivity.json"
ROBUST_JSON="${OUT_DIR}/${PREFIX}_robust_design.json"

if [[ ! -s "${VARIANTS_FASTA}" ]]; then
  echo "missing J sweep variants FASTA: ${VARIANTS_FASTA}" >&2
  exit 1
fi

echo "[start] J=${J_VALUE} gpu=${GPU_INDEX} $(date)"
trap 'status=$?; echo "[exit] J=${J_VALUE} status=${status} $(date)"; exit ${status}' EXIT

CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" prophet/stage2.py \
  --variants-fasta "${VARIANTS_FASTA}" \
  --wt-seq "${WT_SEQ}" \
  --out-json "${DESIGNS_JSON}" \
  --n-designs 500 \
  --n-steps 200 \
  --peptide-length 10 \
  --eta 0.1 \
  --seed 42 \
  --design-mode prophet \
  --affinity-mode peptiverse \
  --peptiverse-normalization raw \
  --device cuda:0 \
  --dfm-device cuda:0 \
  --dfm-ckpt "${DFM_CKPT}"

"${PYTHON}" prophet/eval/pareto.py \
  --designs-json "${DESIGNS_JSON}" \
  --out-json "${PARETO_JSON}"

"${PYTHON}" prophet/eval/eta_sensitivity.py \
  --designs-json "${DESIGNS_JSON}" \
  --out-json "${ETA_JSON}"

CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" prophet/eval/robust_design.py \
  --designs-json "${DESIGNS_JSON}" \
  --wt-seq "${WT_SEQ}" \
  --escape-fasta "${ESCAPE_FASTA}" \
  --out-json "${ROBUST_JSON}" \
  --tau-bind 8.0 \
  --affinity-mode peptiverse \
  --peptiverse-normalization raw \
  --device cuda:0

echo "[done] J=${J_VALUE} $(date)"
