#!/usr/bin/env bash
set -euo pipefail

cd /scratch/pranamlab/kimberly/PROPHET

PYTHON=/scratch/pranamlab/kimberly/PROPHET/venv/bin/python
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ABL_DIR="results/ablations"
MOG_DIR="results/mogdfm_baselines"
HIV_DIR="results/hiv_stage2"
LOG_DIR="results/updated_hiv_reruns_${STAMP}"

VARIANTS_FASTA="results/all_trees_stage1_train_only/hiv_train_gibbs_variants.fasta"
TRAIN_FASTA="data/pre_stage1_split/alignments/train/hiv_train_aligned.fasta"
TEST_FASTA="data/pre_stage1_split/alignments/test/hiv_test_aligned.fasta"
PROPHET_REFERENCE_FASTA="${VARIANTS_FASTA}"
WT_SEQ="PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
DFM_CKPT="MOG-DFM/ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"
N_DESIGNS="${N_DESIGNS:-500}"
N_STEPS="${N_STEPS:-50}"
PEPTIDE_LENGTH="${PEPTIDE_LENGTH:-10}"
BETA="${BETA:-5.0}"
ETA="${ETA:-0.1}"
SEED="${SEED:-42}"
TAU_BIND="${TAU_BIND:-8.0}"

mkdir -p "${ABL_DIR}" "${MOG_DIR}" "${HIV_DIR}" "${LOG_DIR}"

export PYTHONPATH="/scratch/pranamlab/kimberly/PROPHET${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${TORCH_HOME:-/scratch/pranamlab/kimberly/model_cache/torch}"
export HF_HOME="${HF_HOME:-/scratch/pranamlab/kimberly/model_cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/pranamlab/kimberly/model_cache}"
mkdir -p "${TORCH_HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}"

MANIFEST="${LOG_DIR}/manifest.tsv"
printf "name\tpid\tgpu\tlog\tout\n" > "${MANIFEST}"

launch() {
  local name="$1"
  local gpu="$2"
  local log="$3"
  local out="$4"
  shift 4

  echo "[launch] ${name} gpu=${gpu} out=${out}"
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "$@" > "${log}" 2>&1 < /dev/null &
  local pid=$!
  printf "%s\t%s\t%s\t%s\t%s\n" "${name}" "${pid}" "${gpu}" "${log}" "${out}" >> "${MANIFEST}"
}

launch_stage2() {
  local name="$1"
  local gpu="$2"
  local out_json="$3"
  shift 3

  launch "${name}" "${gpu}" "${LOG_DIR}/${name}.log" "${out_json}" \
    "${PYTHON}" prophet/stage2.py \
      --variants-fasta "${VARIANTS_FASTA}" \
      --wt-seq "${WT_SEQ}" \
      --out-json "${out_json}" \
      --n-designs "${N_DESIGNS}" \
      --n-steps "${N_STEPS}" \
      --peptide-length "${PEPTIDE_LENGTH}" \
      --beta "${BETA}" \
      --dfm-ckpt "${DFM_CKPT}" \
      --device cuda:0 \
      --dfm-device cuda:0 \
      --peptiverse-normalization raw \
      --seed "${SEED}" \
      --verbose-sampling \
      "$@"
}

launch_mog() {
  local mode="$1"
  local gpu="$2"
  local prefix="$3"
  shift 3
  local out_json="${MOG_DIR}/${prefix}_stage2_mogdfm_${STAMP}.json"

  launch "mog_${mode}" "${gpu}" "${LOG_DIR}/mog_${mode}.log" "${out_json}" \
    "${PYTHON}" prophet/stage2_mog_baselines.py \
      --variants-fasta "${VARIANTS_FASTA}" \
      --guidance-variants-fasta "${TRAIN_FASTA}" \
      --edit-distance-reference-fasta "${PROPHET_REFERENCE_FASTA}" \
      --guidance-out-fasta "${MOG_DIR}/${prefix}_guidance_variants_${STAMP}.fasta" \
      --wt-seq "${WT_SEQ}" \
      --out-json "${out_json}" \
      --n-designs "${N_DESIGNS}" \
      --n-steps "${N_STEPS}" \
      --peptide-length "${PEPTIDE_LENGTH}" \
      --beta "${BETA}" \
      --seed "${SEED}" \
      --design-mode "${mode}" \
      --dfm-ckpt "${DFM_CKPT}" \
      --device cuda:0 \
      --dfm-device cuda:0 \
      --peptiverse-normalization raw \
      --verbose-sampling \
      "$@"
}

launch_stage2 cvar_eta_1_0 0 "${ABL_DIR}/cvar_eta_1.0-${STAMP}.json" --eta 1.0
launch_stage2 cvar_eta_0_5 1 "${ABL_DIR}/cvar_eta_0.5-${STAMP}.json" --eta 0.5
launch_stage2 cvar_eta_0_1 2 "${ABL_DIR}/cvar_eta_0.1-${STAMP}.json" --eta 0.1
launch_stage2 gibbs_leaves 3 "${ABL_DIR}/gibbs_leaves-${STAMP}.json" --eta "${ETA}" --design-mode uniform_leaves

launch_mog wt_only 4 hiv_train_wt_only
launch_mog uniform_leaves 5 hiv_train_uniform_leaves
launch_mog random_variants 6 hiv_train_random_variants
launch_mog esm_only_variants 7 hiv_train_esm_only_variants
launch_mog prob_weighted_variants 3 hiv_train_prob_weighted_variants \
  --lambda-path results/all_trees_stage1_train_only/hiv_train_lambda.npy \
  --qi-path results/all_trees_stage1_train_only/hiv_train_Qi.npz \
  --h-path results/all_trees_stage1_train_only/hiv_train_h.npy \
  --j-path results/all_trees_stage1_train_only/hiv_train_J.npz \
  --t-evo 1.0 \
  --energy-mode dca_plus_qi

launch peptune_csv_baseline 1 "${LOG_DIR}/peptune_csv_baseline.log" "${HIV_DIR}/hiv_train_peptune_stage2_peptiverse.json" \
  scripts/run_peptune_hiv_baseline.sh 1 "${N_DESIGNS}" "${PEPTIDE_LENGTH}"

launch stage2_peptune_train 2 "${LOG_DIR}/stage2_peptune_train.log" "${HIV_DIR}/hiv_train_stage2_peptune.json" \
  env OUT_DIR="${HIV_DIR}" N_DESIGNS="${N_DESIGNS}" N_STEPS="${N_STEPS}" \
  scripts/run_peptune_stage2_hiv_train.sh 2

echo "[manifest] ${MANIFEST}"
