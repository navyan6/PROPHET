#!/bin/bash
# Generates boot100-boot199 bootstrap trees for J=200 sensitivity sweep (Table 7).
# Prerequisite: boot0-boot99 already exist in
#   data/pre_stage1_split/trees/train/hiv_train_bootstrap/
# After this finishes: sbatch run_stage1_j200.slurm
#
#SBATCH --job-name=hiv_boot200
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hiv_boot200_%j.out
#SBATCH --error=logs/hiv_boot200_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/$USER/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python

cd "$REPO"
mkdir -p logs

ALIGNED=data/pre_stage1_split/alignments/train/hiv_train_aligned.fasta
BOOT_DIR=data/pre_stage1_split/trees/train/hiv_train_bootstrap
TREES_FILE=data/pre_stage1_split/trees/train/hiv_bootstrap_trees_200.txt

echo "Generating boot100-boot199 ..."
$PYTHON build_trees.py \
    "$ALIGNED" \
    --out-dir       data/pre_stage1_split/trees/train \
    --skip-align \
    --n-bootstraps  100 \
    --bootstrap-start 100

echo "Building 200-tree list ..."
# Combine existing 100 trees + new 100 trees into one list
cat data/pre_stage1_split/trees/train/hiv_bootstrap_trees.txt > "$TREES_FILE"
for i in $(seq 100 199); do
    echo "data/pre_stage1_split/trees/train/hiv_train_bootstrap/boot${i}_tree.nwk" >> "$TREES_FILE"
done
echo "Trees list written to $TREES_FILE ($(wc -l < "$TREES_FILE") entries)"
echo "Done. Submit Stage 1 J=200: sbatch run_stage1_j200.slurm"
