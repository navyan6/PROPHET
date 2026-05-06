#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU_LIST [OUT_BASE]" >&2
  echo "example: $0 0,1,2,3 results/hiv_stage2_j_sweep" >&2
  echo "override: J_VALUES='25 50 100 200' $0 0,1,2,3" >&2
  exit 2
fi

GPU_LIST="$1"
OUT_BASE="${2:-results/hiv_stage2_j_sweep}"

cd /scratch/pranamlab/kimberly/PROPHET

PYTHON="${PYTHON:-/scratch/pranamlab/kimberly/PROPHET/venv/bin/python}"
export PYTHONPATH="/scratch/pranamlab/kimberly/PROPHET${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/scratch/pranamlab/kimberly/model_cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/pranamlab/kimberly/model_cache/hf}"
export MOG_DFM_STEP_LOG_EVERY="${MOG_DFM_STEP_LOG_EVERY:-10}"

J_VALUES="${J_VALUES:-1 5 10 25 50 100 144 200}"
J_SWEEP_BASE="${J_SWEEP_BASE:-results/hiv_j_sweep}"
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
read -r -a JS <<< "${J_VALUES}"

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[error] GPU_LIST did not contain any GPU IDs." >&2
  exit 2
fi

mkdir -p "${OUT_BASE}/logs"

run_one_j() {
  local j_value="$1"
  local gpu_index="$2"
  local j_dir="${OUT_BASE}/J_${j_value}"
  local prefix="hiv_train_J_${j_value}_prophet_fixed_omega_050_050"
  local variants_fasta="${J_SWEEP_BASE}/J_${j_value}/hiv_train_J_${j_value}_gibbs_variants.fasta"
  local designs_json="${j_dir}/${prefix}_stage2_peptiverse.json"
  local pareto_json="${j_dir}/${prefix}_pareto.json"
  local eta_json="${j_dir}/${prefix}_eta_sensitivity.json"
  local robust_json="${j_dir}/${prefix}_robust_design.json"

  mkdir -p "${j_dir}"

  if [[ ! -s "${variants_fasta}" ]]; then
    echo "[error] missing variants FASTA for J=${j_value}: ${variants_fasta}" >&2
    return 1
  fi

  echo "[start] J=${j_value} gpu=${gpu_index} out=${j_dir} $(date)"
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

  echo "[done] J=${j_value} gpu=${gpu_index} $(date)"
}

echo "[launcher start] $(date)"
echo "[launcher config] J_VALUES=${J_VALUES}"
echo "[launcher config] GPUS=${GPU_LIST}"
echo "[launcher config] OUT_BASE=${OUT_BASE}"
echo "[launcher config] omega=(${OMEGA_BINDING_WEIGHT}, $("${PYTHON}" -c "w=float('${OMEGA_BINDING_WEIGHT}'); print(f'{1.0-w:.3f}')"))"

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LOGS=()
declare -a ACTIVE_JS=()
status=0

for idx in "${!JS[@]}"; do
  j_value="${JS[$idx]}"
  gpu_index="${GPUS[$((idx % ${#GPUS[@]}))]}"
  log="${OUT_BASE}/logs/J_${j_value}_gpu${gpu_index}.log"

  echo "[launch] J=${j_value} gpu=${gpu_index} log=${log}"
  run_one_j "${j_value}" "${gpu_index}" > "${log}" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LOGS+=("${log}")
  ACTIVE_JS+=("${j_value}")

  if [[ "${#ACTIVE_PIDS[@]}" -ge "${#GPUS[@]}" ]]; then
    job_status=0
    wait "${ACTIVE_PIDS[0]}" || job_status=$?
    if [[ "${job_status}" -ne 0 ]]; then
      status=1
    fi
    echo "[finish] J=${ACTIVE_JS[0]} status=${job_status} log=${ACTIVE_LOGS[0]}"
    ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
    ACTIVE_LOGS=("${ACTIVE_LOGS[@]:1}")
    ACTIVE_JS=("${ACTIVE_JS[@]:1}")
  fi
done

for idx in "${!ACTIVE_PIDS[@]}"; do
  job_status=0
  wait "${ACTIVE_PIDS[$idx]}" || job_status=$?
  if [[ "${job_status}" -ne 0 ]]; then
    status=1
  fi
  echo "[finish] J=${ACTIVE_JS[$idx]} status=${job_status} log=${ACTIVE_LOGS[$idx]}"
done

if [[ "${status}" -ne 0 ]]; then
  echo "[launcher done] one or more J jobs failed; check ${OUT_BASE}/logs" >&2
  exit "${status}"
fi

echo "[launcher done] all J jobs complete $(date)"
