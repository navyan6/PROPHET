#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${1:-0}"

ROOT=/scratch/pranamlab/kimberly/PROPHET
PYTHON="${ROOT}/venv/bin/python"
OUT_DIR="${OUT_DIR:-${ROOT}/results/hiv_stage2}"
VARIANTS_FASTA="${VARIANTS_FASTA:-${ROOT}/results/all_trees_stage1_train_only/hiv_train_gibbs_variants.fasta}"
WT_SEQ="${WT_SEQ:-PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF}"
CKPT="${CKPT:-${ROOT}/PepTune/checkpoints/peptune-pretrained.ckpt}"
N_DESIGNS="${N_DESIGNS:-500}"
N_STEPS="${N_STEPS:-50}"
SEQ_LENGTH="${SEQ_LENGTH:-10}"
PEPTUNE_CHILDREN="${PEPTUNE_CHILDREN:-50}"
ETA="${ETA:-0.1}"
SEED="${SEED:-42}"
TAU_BIND="${TAU_BIND:-8.0}"

mkdir -p "${OUT_DIR}"

export TORCH_HOME="${TORCH_HOME:-/scratch/pranamlab/kimberly/model_cache/torch}"
export HF_HOME="${HF_HOME:-/scratch/pranamlab/kimberly/model_cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/pranamlab/kimberly/model_cache}"
mkdir -p "${TORCH_HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}"

if [[ ! -s "${CKPT}" ]]; then
  echo "[error] Missing PepTune checkpoint: ${CKPT}" >&2
  exit 1
fi

DESIGNS_JSON="${OUT_DIR}/hiv_train_stage2_peptune.json"

echo "[start] stage2_peptune HIV train gpu=${GPU_INDEX} $(date)"
cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" prophet/stage2_peptune.py \
  --variants-fasta "${VARIANTS_FASTA}" \
  --wt-seq "${WT_SEQ}" \
  --out-json "${DESIGNS_JSON}" \
  --eta "${ETA}" \
  --tau-bind "${TAU_BIND}" \
  --peptiverse-normalization raw \
  --device cuda:0 \
  --seed "${SEED}" \
  --ckpt "${CKPT}" \
  --seq-length "${SEQ_LENGTH}" \
  --sampling-steps "${N_STEPS}" \
  --num-iter "${N_DESIGNS}" \
  --num-children "${PEPTUNE_CHILDREN}"

"${PYTHON}" prophet/eval/pareto.py \
  --designs-json "${DESIGNS_JSON}" \
  --out-json "${OUT_DIR}/hiv_train_stage2_peptune_pareto.json"

"${PYTHON}" prophet/eval/eta_sensitivity.py \
  --designs-json "${DESIGNS_JSON}" \
  --out-json "${OUT_DIR}/hiv_train_stage2_peptune_eta_sensitivity.json"

echo "[done] stage2_peptune HIV train $(date)"
