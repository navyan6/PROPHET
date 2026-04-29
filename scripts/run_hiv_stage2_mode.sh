#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 MODE GPU_INDEX" >&2
  exit 2
fi

MODE="$1"
GPU_INDEX="$2"

cd /scratch/pranamlab/kimberly/PROPHET

PYTHON=/scratch/pranamlab/kimberly/PROPHET/venv/bin/python
OUT_DIR=results/hiv_stage2
VARIANTS_FASTA=results/all_trees_stage1_train_only/hiv_train_gibbs_variants.fasta
ESCAPE_FASTA=data/pre_stage1_split/alignments/test/hiv_test_aligned.fasta
WT_SEQ=PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF
DFM_CKPT=MOG-DFM/ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt

mkdir -p "${OUT_DIR}"

DESIGNS_JSON="${OUT_DIR}/hiv_train_${MODE}_stage2_peptiverse.json"
PARETO_JSON="${OUT_DIR}/hiv_train_${MODE}_pareto.json"
ETA_JSON="${OUT_DIR}/hiv_train_${MODE}_eta_sensitivity.json"
ROBUST_JSON="${OUT_DIR}/hiv_train_${MODE}_robust_design.json"

echo "[start] mode=${MODE} gpu=${GPU_INDEX} $(date)"

CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" prophet/stage2.py \
  --variants-fasta "${VARIANTS_FASTA}" \
  --wt-seq "${WT_SEQ}" \
  --out-json "${DESIGNS_JSON}" \
  --n-designs 500 \
  --n-steps 200 \
  --peptide-length 10 \
  --eta 0.1 \
  --seed 42 \
  --design-mode "${MODE}" \
  --affinity-mode peptiverse \
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
  --tau-bind 0.5 \
  --affinity-mode peptiverse \
  --device cuda:0

echo "[done] mode=${MODE} $(date)"
