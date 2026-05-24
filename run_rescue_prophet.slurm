#!/bin/bash
#SBATCH --job-name=hiv_s2_prophet
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:B200:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hiv_s2_prophet_%j.out
#SBATCH --error=logs/hiv_s2_prophet_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/nnori/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
cd "$REPO"
mkdir -p logs results/ablations

WT_SEQ="PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
VARIANTS="results/hiv_prophet_final/hiv_prophet_t015_esm_filtered_d20.fasta"
STAMP=$(date +%Y%m%d_%H%M%S)

$PYTHON prophet/stage2.py \
    --variants-fasta  "$VARIANTS" \
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
    --out-json        "results/ablations/t2_prophet_rescue-${STAMP}.json" \
    --verbose-sampling
