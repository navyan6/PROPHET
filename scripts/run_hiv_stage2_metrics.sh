#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 MODE DESIGNS_JSON GPU_INDEX" >&2
  exit 2
fi

MODE="$1"
DESIGNS_JSON="$2"
GPU_INDEX="$3"

cd /scratch/pranamlab/kimberly/PROPHET

PYTHON=/scratch/pranamlab/kimberly/PROPHET/venv/bin/python
OUT_DIR="${OUT_DIR:-results/hiv_stage2}"
ESCAPE_FASTA=data/pre_stage1_split/alignments/test/hiv_test_aligned.fasta
WT_SEQ=PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF

PARETO_JSON="${OUT_DIR}/hiv_train_${MODE}_pareto.json"
ETA_JSON="${OUT_DIR}/hiv_train_${MODE}_eta_sensitivity.json"
ROBUST_JSON="${OUT_DIR}/hiv_train_${MODE}_robust_design.json"

echo "[metrics-start] mode=${MODE} designs=${DESIGNS_JSON} $(date)"

until [[ -s "${DESIGNS_JSON}" ]]; do
  sleep 60
done

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

echo "[metrics-done] mode=${MODE} $(date)"
