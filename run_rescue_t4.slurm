#!/bin/bash
# Rescue jobs for Table 4 Stage-1 ablations: no_dca, no_lambda, no_esm.
# Runs all 3 in parallel on separate GPUs with --variant-limit 149.
#
#SBATCH --job-name=hiv_s2_t4
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:B200:3
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/hiv_s2_t4_%j.out
#SBATCH --error=logs/hiv_s2_t4_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/nnori/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
export HF_HOME=/vast/projects/pranam/lab/nnori/.cache/huggingface

cd "$REPO"
mkdir -p logs results/ablations

WT_SEQ="PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
STAMP=$(date +%Y%m%d_%H%M%S)

run_exp() {
    local name=$1
    local fasta=$2
    local gpu=$3
    if [ ! -f "$fasta" ]; then
        echo "[skip] $name: $fasta not found"
        return
    fi
    echo "Launching $name on GPU $gpu ..."
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON prophet/stage2.py \
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
        > "results/ablations/${name}-${STAMP}.log" 2>&1 &
}

run_exp t4_no_dca    results/stage1_ablations/hiv_no_dca_gibbs_variants.fasta    0
run_exp t4_no_lambda results/stage1_ablations/hiv_no_lambda_gibbs_variants.fasta 1
run_exp t4_no_esm    results/stage1_ablations/hiv_no_esm_gibbs_variants.fasta    2

echo "All 3 t4 ablation jobs launched. Waiting ..."
wait
echo "All t4 ablation jobs complete."
