#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU_LIST [OUT_BASE]" >&2
  echo "example: $0 0,1,2,3 results/hiv_stage2_m_sweep" >&2
  echo "override: M_VALUES='50 100 250 500' $0 0,1,2,3" >&2
  exit 2
fi

GPU_LIST="$1"
OUT_BASE="${2:-results/hiv_stage2_m_sweep}"

cd /scratch/pranamlab/kimberly/PROPHET

PYTHON="${PYTHON:-/scratch/pranamlab/kimberly/PROPHET/venv/bin/python}"
export PYTHONPATH="/scratch/pranamlab/kimberly/PROPHET${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/scratch/pranamlab/kimberly/model_cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/pranamlab/kimberly/model_cache/hf}"
export MOG_DFM_STEP_LOG_EVERY="${MOG_DFM_STEP_LOG_EVERY:-10}"

M_VALUES="${M_VALUES:-50 100 250 500}"
M_SWEEP_BASE="${M_SWEEP_BASE:-results/hiv_m_sweep}"
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

GPU_LIST="${GPU_LIST//,/ }"
read -r -a GPUS <<< "${GPU_LIST}"
read -r -a MS <<< "${M_VALUES}"

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[error] GPU_LIST did not contain any GPU IDs." >&2
  exit 2
fi

mkdir -p "${OUT_BASE}/logs"

run_one_m() {
  local m_value="$1"
  local gpu_index="$2"
  local m_dir="${OUT_BASE}/M_${m_value}"
  local prefix="hiv_train_M_${m_value}_prophet_fixed_omega_050_050"
  local variants_fasta="${M_SWEEP_BASE}/M_${m_value}/hiv_train_M_${m_value}_gibbs_variants.fasta"
  local designs_json="${m_dir}/${prefix}_stage2_peptiverse.json"
  local pareto_json="${m_dir}/${prefix}_pareto.json"
  local eta_json="${m_dir}/${prefix}_eta_sensitivity.json"
  local robust_json="${m_dir}/${prefix}_robust_design.json"

  mkdir -p "${m_dir}"

  if [[ ! -s "${variants_fasta}" ]]; then
    echo "[error] missing variants FASTA for M=${m_value}: ${variants_fasta}" >&2
    return 1
  fi

  echo "[start] M=${m_value} gpu=${gpu_index} out=${m_dir} $(date)"
  echo "[input] ${variants_fasta}"
  echo "[json] designs=${designs_json}"
  echo "[json] pareto=${pareto_json}"
  echo "[json] eta=${eta_json}"
  echo "[json] robust=${robust_json}"

  CUDA_VISIBLE_DEVICES="${gpu_index}" PYTHONUNBUFFERED=1 "${PYTHON}" prophet/stage2.py \
    --variants-fasta "${variants_fasta}" \
    --wt-seq "${WT_SEQ}" \
    --out-json "${designs_json}" \
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
    --designs-json "${designs_json}" \
    --out-json "${pareto_json}"

  "${PYTHON}" prophet/eval/eta_sensitivity.py \
    --designs-json "${designs_json}" \
    --out-json "${eta_json}"

  CUDA_VISIBLE_DEVICES="${gpu_index}" PYTHONUNBUFFERED=1 "${PYTHON}" prophet/eval/robust_design.py \
    --designs-json "${designs_json}" \
    --wt-seq "${WT_SEQ}" \
    --escape-fasta "${ESCAPE_FASTA}" \
    --out-json "${robust_json}" \
    --tau-bind "${TAU_BIND}" \
    --affinity-mode peptiverse \
    --peptiverse-normalization raw \
    --device cuda:0

  echo "[done] M=${m_value} gpu=${gpu_index} $(date)"
}

echo "[launcher start] $(date)"
echo "[launcher config] M_VALUES=${M_VALUES}"
echo "[launcher config] GPUS=${GPU_LIST}"
echo "[launcher config] OUT_BASE=${OUT_BASE}"
echo "[launcher config] omega=(${OMEGA_BINDING_WEIGHT}, $("${PYTHON}" -c "w=float('${OMEGA_BINDING_WEIGHT}'); print(f'{1.0-w:.3f}')"))"

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LOGS=()
declare -a ACTIVE_MS=()
status=0

wait_for_oldest() {
  local job_status=0

  wait "${ACTIVE_PIDS[0]}" || job_status=$?
  if [[ "${job_status}" -ne 0 ]]; then
    status=1
  fi

  echo "[finish] M=${ACTIVE_MS[0]} status=${job_status} log=${ACTIVE_LOGS[0]}"
  ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
  ACTIVE_LOGS=("${ACTIVE_LOGS[@]:1}")
  ACTIVE_MS=("${ACTIVE_MS[@]:1}")
}

for idx in "${!MS[@]}"; do
  m_value="${MS[$idx]}"
  gpu_index="${GPUS[$((idx % ${#GPUS[@]}))]}"
  log="${OUT_BASE}/logs/M_${m_value}_gpu${gpu_index}.log"

  echo "[launch] M=${m_value} gpu=${gpu_index} log=${log}"
  run_one_m "${m_value}" "${gpu_index}" > "${log}" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LOGS+=("${log}")
  ACTIVE_MS+=("${m_value}")

  if [[ "${#ACTIVE_PIDS[@]}" -ge "${#GPUS[@]}" ]]; then
    wait_for_oldest
  fi
done

while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
  wait_for_oldest
done

if [[ "${status}" -ne 0 ]]; then
  echo "[launcher done] one or more M jobs failed; check ${OUT_BASE}/logs" >&2
  exit "${status}"
fi

echo "[launcher done] all M jobs complete $(date)"
