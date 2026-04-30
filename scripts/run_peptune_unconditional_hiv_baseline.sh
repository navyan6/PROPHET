#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${1:-0}"
N_DESIGNS="${2:-500}"
SEQ_LENGTH="${3:-10}"

ROOT=/scratch/pranamlab/kimberly/PROPHET
PYTHON="${ROOT}/venv/bin/python"
PEPTUNE_DIR="${ROOT}/PepTune"
OUT_DIR="${PEPTUNE_OUT_DIR:-${ROOT}/results/peptune_hiv}"
STAGE2_OUT="${STAGE2_OUT:-${ROOT}/results/hiv_stage2}"
WT_SEQ=PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF
TEST_FASTA="${ROOT}/data/pre_stage1_split/alignments/test/hiv_test_aligned.fasta"
CKPT="${PEPTUNE_DIR}/checkpoints/peptune-pretrained.ckpt"
CSV="${OUT_DIR}/hiv_train_peptune_unconditional.csv"

mkdir -p "${OUT_DIR}" "${STAGE2_OUT}"

if [[ ! -s "${CKPT}" ]]; then
  echo "[error] Missing PepTune checkpoint: ${CKPT}" >&2
  echo "Download peptune-pretrained.ckpt into PepTune/checkpoints/ before running this baseline." >&2
  exit 1
fi

echo "[start] PepTune unconditional HIV baseline gpu=${GPU_INDEX} $(date)"
pushd "${PEPTUNE_DIR}/src" >/dev/null
MPLBACKEND=Agg \
PEPTUNE_BASE_PATH="${PEPTUNE_DIR}" \
PEPTUNE_CKPT_PATH="${CKPT}" \
PEPTUNE_TARGET_SEQ="${WT_SEQ}" \
PEPTUNE_OUTPUT_CSV="${CSV}" \
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
PYTHONUNBUFFERED=1 \
"${PYTHON}" generate_unconditional.py \
  base_path="${PEPTUNE_DIR}" \
  sampling.num_sequences="${N_DESIGNS}" \
  sampling.seq_length="${SEQ_LENGTH}" \
  sampling.steps=128 \
  hydra.run.dir=. \
  hydra.job.chdir=False
popd >/dev/null

DESIGNS_JSON="${STAGE2_OUT}/hiv_train_peptune_unconditional_stage2_peptiverse.json"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" "${ROOT}/scripts/evaluate_generated_peptides.py" \
  --input "${CSV}" \
  --input-format csv \
  --method peptune_unconditional \
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
  --out-json "${STAGE2_OUT}/hiv_train_peptune_unconditional_pareto.json"

"${PYTHON}" "${ROOT}/prophet/eval/eta_sensitivity.py" \
  --designs-json "${DESIGNS_JSON}" \
  --out-json "${STAGE2_OUT}/hiv_train_peptune_unconditional_eta_sensitivity.json"

echo "[done] PepTune unconditional HIV baseline $(date)"
