#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${1:-0}"
N_DESIGNS="${2:-500}"
SEQ_LENGTH="${3:-10}"

ROOT=/scratch/pranamlab/kimberly/PROPHET
PYTHON="${ROOT}/venv/bin/python"
RFDIR="${ROOT}/RFdiffusion"
OUT_DIR="${RFDIFFUSION_OUT_DIR:-${ROOT}/results/rfdiffusion_hiv}"
STAGE2_OUT="${STAGE2_OUT:-${ROOT}/results/hiv_stage2}"
WT_SEQ=PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF
TEST_FASTA="${ROOT}/data/pre_stage1_split/alignments/test/hiv_test_aligned.fasta"

mkdir -p "${OUT_DIR}" "${STAGE2_OUT}"

if [[ ! -s "${RFDIR}/models/Base_ckpt.pt" ]]; then
  echo "[error] Missing RFdiffusion Base checkpoint: ${RFDIR}/models/Base_ckpt.pt" >&2
  echo "Run RFdiffusion/scripts/download_models.sh or place RFdiffusion weights before running." >&2
  exit 1
fi

echo "[start] RFdiffusion HIV unconditional baseline gpu=${GPU_INDEX} $(date)"
pushd "${RFDIR}" >/dev/null
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" scripts/run_inference.py \
  inference.output_prefix="${OUT_DIR}/hiv_rf" \
  inference.num_designs="${N_DESIGNS}" \
  inference.write_trajectory=False \
  "contigmap.contigs=[${SEQ_LENGTH}-${SEQ_LENGTH}]"
popd >/dev/null

DESIGNS_JSON="${STAGE2_OUT}/hiv_train_rfdiffusion_stage2_peptiverse.json"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" "${ROOT}/scripts/evaluate_generated_peptides.py" \
  --input "${OUT_DIR}" \
  --input-format pdb_dir \
  --method rfdiffusion \
  --wt-seq "${WT_SEQ}" \
  --test-fasta "${TEST_FASTA}" \
  --out-json "${DESIGNS_JSON}" \
  --eta 0.1 \
  --tau-bind 8.0 \
  --device cuda:0 \
  --peptiverse-normalization raw \
  --dedupe

"${PYTHON}" "${ROOT}/prophet/eval/pareto.py" \
  --designs-json "${DESIGNS_JSON}" \
  --out-json "${STAGE2_OUT}/hiv_train_rfdiffusion_pareto.json"

"${PYTHON}" "${ROOT}/prophet/eval/eta_sensitivity.py" \
  --designs-json "${DESIGNS_JSON}" \
  --out-json "${STAGE2_OUT}/hiv_train_rfdiffusion_eta_sensitivity.json"

echo "[done] RFdiffusion HIV baseline $(date)"
