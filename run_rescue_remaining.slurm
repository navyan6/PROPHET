#!/bin/bash
# Rescue job for experiments that may not finish in hiv_all_stage2 (6076907).
# Covers: t2_uniform_leaves, t5_eta_0.1, t5_eta_1.0, t7_J_25, t7_J_50, t7_J_100
# Skips any experiment that already has a JSON saved in results/ablations/.
#
#SBATCH --job-name=hiv_s2_rescue
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:B200:6
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=logs/hiv_s2_rescue_%j.out
#SBATCH --error=logs/hiv_s2_rescue_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/nnori/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
export HF_HOME=/vast/projects/pranam/lab/nnori/.cache/huggingface

cd "$REPO"
mkdir -p logs results/ablations

WT_SEQ="PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
VARIANTS="results/hiv_prophet_final/hiv_prophet_t015_esm_filtered_d20.fasta"
STAMP=$(date +%Y%m%d_%H%M%S)

# Returns 0 if a JSON for this tag already exists (skip it), 1 if missing (run it)
needs_run() {
    local tag=$1
    compgen -G "results/ablations/${tag}-*.json" > /dev/null 2>&1 && return 1 || return 0
}

run_exp() {
    local name=$1
    local fasta=$2
    local gpu=$3
    local extra=$4
    if ! needs_run "$name"; then
        echo "[skip] $name: JSON already exists"
        return
    fi
    if [ ! -f "$fasta" ]; then
        echo "[skip] $name: variant FASTA not found at $fasta"
        return
    fi
    echo "Launching $name on GPU $gpu ..."
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PYTHON prophet/stage2.py \
        --variants-fasta  "$fasta" \
        --wt-seq          "$WT_SEQ" \
        --device          cuda:0 \
        --dfm-device      cuda:0 \
        --n-designs       500 \
        --n-steps         200 \
        --peptide-length  10 \
        --beta            5.0 \
        --eta             0.1 \
        --tau-bind        7.5 \
        --peptiverse-normalization raw \
        --variant-limit   149 \
        --out-json        "results/ablations/${name}-${STAMP}.json" \
        --verbose-sampling \
        $extra \
        > "results/ablations/${name}-${STAMP}.log" 2>&1 &
}

# GPU assignments (names must match make_paper_tables.py tag lookup)
run_exp "t2_uniform_leaves" "$VARIANTS"                                                       0 "--design-mode uniform_leaves"
run_exp "t5_eta_0.1"        "$VARIANTS"                                                       1 "--eta 0.1"
run_exp "t5_eta_1.0"        "$VARIANTS"                                                       2 "--eta 1.0"
run_exp "t7_J_25"           "results/stage1_j_sweep/J_25/hiv_train_gibbs_variants.fasta"     3 ""
run_exp "t7_J_50"           "results/stage1_j_sweep/J_50/hiv_train_gibbs_variants.fasta"     4 ""
run_exp "t7_J_100"          "$VARIANTS"                                                       5 ""

echo "All rescue jobs launched. Waiting for completion ..."
wait
echo "All rescue jobs complete."
