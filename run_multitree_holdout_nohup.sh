#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   nohup bash run_multitree_holdout_nohup.sh > logs/multitree_holdout.nohup.log 2>&1 &

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs data/trees data/results/multi_tree_eval

echo "[INFO] exporting COVID tree bundle..."
python tree_analysis/src/export_generic_hadsbm_tree.py \
  --nwk covid_tree_gen/covid_tree.nwk \
  --fasta covid_tree_gen/covdata.fasta \
  --out data/trees/covid_hadsbm_tree.json \
  --prob-mode length

echo "[INFO] exporting flu tree bundle..."
python tree_analysis/src/export_generic_hadsbm_tree.py \
  --nwk flu_tree/ha_tree.nwk \
  --fasta flu_tree/ha_sampled.fasta \
  --out data/trees/flu_hadsbm_tree.json \
  --prob-mode length

echo "[INFO] running multi-tree holdout evaluation..."
python peptide_optimization/src/multi_tree_holdout_eval.py \
  --tree-jsons \
    data/trees/hadsbm_tree.json \
    data/trees/DENV3_hadsbm_tree.json \
    data/trees/covid_hadsbm_tree.json \
    data/trees/flu_hadsbm_tree.json \
  --labels hiv dengue covid flu \
  --num-candidates 500 \
  --select-top-k 100 \
  --holdout-fraction 0.2 \
  --split-seed 1986 \
  --length 12 \
  --retention-threshold 5.0 \
  --device cuda:0 \
  --out-dir data/results/multi_tree_eval

echo "[DONE] summary at data/results/multi_tree_eval/summary.csv"

