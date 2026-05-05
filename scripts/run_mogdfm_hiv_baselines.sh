#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU_INDEX [OUT_DIR]" >&2
  echo "example: $0 0 results/hiv_mogdfm_baselines" >&2
  exit 2
fi

GPU_INDEX="$1"
OUT_DIR="${2:-results/hiv_mogdfm_baselines}"

cd /scratch/pranamlab/kimberly/PROPHET

PYTHON="${PYTHON:-/scratch/pranamlab/kimberly/PROPHET/venv/bin/python}"
export PYTHONPATH="/scratch/pranamlab/kimberly/PROPHET${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_FASTA="${TRAIN_FASTA:-data/pre_stage1_split/alignments/train/hiv_train_aligned.fasta}"
TEST_FASTA="${TEST_FASTA:-data/pre_stage1_split/alignments/test/hiv_test_aligned.fasta}"
PROPHET_REFERENCE_FASTA="${PROPHET_REFERENCE_FASTA:-results/all_trees_stage1_train_only/hiv_train_gibbs_variants.fasta}"
WT_SEQ="${WT_SEQ:-PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF}"
DFM_CKPT="${DFM_CKPT:-MOG-DFM/ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt}"
N_DESIGNS="${N_DESIGNS:-500}"
N_STEPS="${N_STEPS:-50}"
PEPTIDE_LENGTH="${PEPTIDE_LENGTH:-10}"
ETA="${ETA:-0.1}"
SEED="${SEED:-42}"
TAU_BIND="${TAU_BIND:-8.0}"
GUIDANCE_VAR_LIMIT="${GUIDANCE_VAR_LIMIT:-50}"
ESM_VARIANT_MODEL="${ESM_VARIANT_MODEL:-facebook/esm2_t6_8M_UR50D}"
ESM_VARIANT_TEMPERATURE="${ESM_VARIANT_TEMPERATURE:-1.0}"

mkdir -p "${OUT_DIR}"

run_mode() {
  local mode="$1"
  local prefix="$2"
  local designs_json="${OUT_DIR}/${prefix}_stage2_peptiverse.json"
  local pareto_json="${OUT_DIR}/${prefix}_pareto.json"
  local eta_json="${OUT_DIR}/${prefix}_eta_sensitivity.json"
  local robust_json="${OUT_DIR}/${prefix}_robust_design.json"
  local guidance_fasta="${OUT_DIR}/${prefix}_guidance_variants.fasta"

  echo "[start] ${mode} gpu=${GPU_INDEX} $(date)"

  CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" prophet/stage2.py \
    --variants-fasta "${TRAIN_FASTA}" \
    --guidance-variants-fasta "${TRAIN_FASTA}" \
    --edit-distance-reference-fasta "${PROPHET_REFERENCE_FASTA}" \
    --guidance-out-fasta "${guidance_fasta}" \
    --wt-seq "${WT_SEQ}" \
    --out-json "${designs_json}" \
    --n-designs "${N_DESIGNS}" \
    --n-steps "${N_STEPS}" \
    --peptide-length "${PEPTIDE_LENGTH}" \
    --eta "${ETA}" \
    --seed "${SEED}" \
    --design-mode "${mode}" \
    --affinity-mode peptiverse \
    --peptiverse-normalization raw \
    --tau-bind "${TAU_BIND}" \
    --guidance-var-limit "${GUIDANCE_VAR_LIMIT}" \
    --device cuda:0 \
    --dfm-device cuda:0 \
    --dfm-ckpt "${DFM_CKPT}" \
    --esm-variant-model "${ESM_VARIANT_MODEL}" \
    --esm-variant-device cuda:0 \
    --esm-variant-temperature "${ESM_VARIANT_TEMPERATURE}"

  "${PYTHON}" prophet/eval/pareto.py \
    --designs-json "${designs_json}" \
    --out-json "${pareto_json}"

  "${PYTHON}" prophet/eval/eta_sensitivity.py \
    --designs-json "${designs_json}" \
    --out-json "${eta_json}"

  CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" prophet/eval/robust_design.py \
    --designs-json "${designs_json}" \
    --wt-seq "${WT_SEQ}" \
    --escape-fasta "${TEST_FASTA}" \
    --out-json "${robust_json}" \
    --tau-bind "${TAU_BIND}" \
    --affinity-mode peptiverse \
    --peptiverse-normalization raw \
    --device cuda:0

  echo "[done] ${mode} $(date)"
}

run_mode wt_only mogdfm_wt_only
run_mode uniform_leaves mogdfm_uniform_leaves
run_mode random_variants mogdfm_random_variants
run_mode esm_only_variants mogdfm_esm_only_variants
