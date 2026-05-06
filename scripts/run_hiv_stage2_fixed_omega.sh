#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU_INDEX [OUT_DIR]" >&2
  exit 2
fi

GPU_INDEX="$1"
OUT_DIR="${2:-results/hiv_stage2_fixed_omega}"

cd /scratch/pranamlab/kimberly/PROPHET

PYTHON="${PYTHON:-/scratch/pranamlab/kimberly/PROPHET/venv/bin/python}"
export PYTHONPATH="/scratch/pranamlab/kimberly/PROPHET${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/scratch/pranamlab/kimberly/model_cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/pranamlab/kimberly/model_cache/hf}"
export MOG_DFM_STEP_LOG_EVERY="${MOG_DFM_STEP_LOG_EVERY:-10}"

VARIANTS_FASTA="${VARIANTS_FASTA:-results/all_trees_stage1_train_only/hiv_train_gibbs_variants.fasta}"
ESCAPE_FASTA="${ESCAPE_FASTA:-data/pre_stage1_split/alignments/test/hiv_test_aligned.fasta}"
WT_SEQ="${WT_SEQ:-PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF}"
DFM_CKPT="${DFM_CKPT:-MOG-DFM/ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt}"
N_DESIGNS="${N_DESIGNS:-500}"
N_STEPS="${N_STEPS:-50}"
PEPTIDE_LENGTH="${PEPTIDE_LENGTH:-10}"
ETA="${ETA:-0.1}"
SEED="${SEED:-42}"
TAU_BIND="${TAU_BIND:-8.0}"
GUIDANCE_VAR_LIMIT="${GUIDANCE_VAR_LIMIT:-50}"
OMEGA_BINDING_WEIGHT="${OMEGA_BINDING_WEIGHT:-0.5}"

mkdir -p "${OUT_DIR}"

PREFIX="hiv_train_prophet_fixed_omega_050_050"
DESIGNS_JSON="${OUT_DIR}/${PREFIX}_stage2_peptiverse.json"
PARETO_JSON="${OUT_DIR}/${PREFIX}_pareto.json"
ETA_JSON="${OUT_DIR}/${PREFIX}_eta_sensitivity.json"
ROBUST_JSON="${OUT_DIR}/${PREFIX}_robust_design.json"

echo "[start] fixed omega run gpu=${GPU_INDEX} omega=(${OMEGA_BINDING_WEIGHT},$(python3 - <<PY
w = float("${OMEGA_BINDING_WEIGHT}")
print(f"{1.0 - w:.1f}")
PY
)) $(date)"

CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" prophet/stage2.py \
  --variants-fasta "${VARIANTS_FASTA}" \
  --wt-seq "${WT_SEQ}" \
  --out-json "${DESIGNS_JSON}" \
  --n-designs "${N_DESIGNS}" \
  --n-steps "${N_STEPS}" \
  --peptide-length "${PEPTIDE_LENGTH}" \
  --eta "${ETA}" \
  --seed "${SEED}" \
  --design-mode prophet \
  --peptiverse-normalization raw \
  --tau-bind "${TAU_BIND}" \
  --guidance-var-limit "${GUIDANCE_VAR_LIMIT}" \
  --device cuda:0 \
  --dfm-device cuda:0 \
  --dfm-ckpt "${DFM_CKPT}" \
  --omega-binding-weight "${OMEGA_BINDING_WEIGHT}"

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
  --tau-bind "${TAU_BIND}" \
  --affinity-mode peptiverse \
  --peptiverse-normalization raw \
  --device cuda:0

echo "[done] fixed omega run $(date)"
