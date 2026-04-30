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

mkdir -p "${OUT_DIR}" "${STAGE2_OUT}"

export TORCH_HOME="${TORCH_HOME:-/scratch/pranamlab/kimberly/model_cache/torch}"
export HF_HOME="${HF_HOME:-/scratch/pranamlab/kimberly/model_cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/pranamlab/kimberly/model_cache}"
mkdir -p "${TORCH_HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}"

if [[ ! -s "${CKPT}" ]]; then
  echo "[error] Missing PepTune checkpoint: ${CKPT}" >&2
  echo "Download peptune-pretrained.ckpt into PepTune/checkpoints/ before running this baseline." >&2
  exit 1
fi

echo "[start] PepTune HIV train-target baseline gpu=${GPU_INDEX} $(date)"

pushd "${PEPTUNE_DIR}/src" >/dev/null
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" generate_mcts.py \
  base_path="${PEPTUNE_DIR}" \
  eval.checkpoint_path="${CKPT}" \
  +prot_name1=hiv_train \
  +prot_seq1="${WT_SEQ}" \
  mode=2 \
  +model_type=mcts \
  +length="${SEQ_LENGTH}" \
  +epoch=0 \
  sampling.seq_length="${SEQ_LENGTH}" \
  sampling.steps=128 \
  mcts.num_iter="${N_DESIGNS}" \
  mcts.num_children=50 \
  hydra.run.dir=. \
  hydra.job.chdir=False
popd >/dev/null

CSV="${PEPTUNE_DIR}/hiv_train/2_mcts_length_${SEQ_LENGTH}_epoch_0.csv"
DESIGNS_JSON="${STAGE2_OUT}/hiv_train_peptune_stage2_peptiverse.json"

CUDA_VISIBLE_DEVICES="${GPU_INDEX}" PYTHONUNBUFFERED=1 "${PYTHON}" "${ROOT}/scripts/evaluate_generated_peptides.py" \
  --input "${CSV}" \
  --input-format csv \
  --method peptune \
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
  --out-json "${STAGE2_OUT}/hiv_train_peptune_pareto.json"

"${PYTHON}" "${ROOT}/prophet/eval/eta_sensitivity.py" \
  --designs-json "${DESIGNS_JSON}" \
  --out-json "${STAGE2_OUT}/hiv_train_peptune_eta_sensitivity.json"

echo "[done] PepTune HIV baseline $(date)"
