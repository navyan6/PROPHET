#!/bin/bash
#SBATCH --job-name=dengue_mogdfm
#SBATCH --account=gts-YOURACCOUNTID-coda20      # ← CHANGE: your charge account (e.g. gts-abc123)
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dengue_%j.out
#SBATCH --error=logs/dengue_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUREMAIL@gatech.edu         # ← CHANGE: your email

# ── paths — CHANGE THIS ONE LINE ─────────────────────────────────────────────
REPO=$HOME/hadsbm-hiv          # ← CHANGE if you cloned elsewhere

# ── conda activation (ICE-specific) ──────────────────────────────────────────
module purge
module load anaconda3
source activate hadsbm          # 'conda activate' doesn't work inside SLURM; use source

cd "$REPO"

echo "=============================="
echo "Job        : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Python     : $(which python)"
echo "Working dir: $(pwd)"
echo "=============================="

# ── Stage 0: ensure dengue JSON exists ───────────────────────────────────────
DENGUE_JSON="$REPO/data/trees/DENV3_hadsbm_tree.json"
mkdir -p "$REPO/data/trees" "$REPO/results/dengue" "$REPO/logs"

if [ ! -f "$DENGUE_JSON" ]; then
    echo "[Stage 0] Generating DENV3 hadsbm JSON..."
    python dengue_tree_gen/dengue_hadsbm_export.py \
        --prob-mode uniform \
        --out "$DENGUE_JSON"
    echo "[Stage 0] Done: $DENGUE_JSON"
else
    echo "[Stage 0] JSON exists: $DENGUE_JSON"
fi

# ── Stage 1: baseline (random peptides + PeptiVerse binding) ─────────────────
echo ""
echo "[Stage 1] Random-peptide baseline..."

python peptide_optimization/src/binding_affinity_simple.py \
    --tree-json   "$DENGUE_JSON" \
    --num-peptides 20 \
    --length 12 \
    --top-k 20 \
    --device cuda:0 \
    --output "$REPO/results/dengue/baseline_${SLURM_JOB_ID}.json"

echo "[Stage 1] Done: results/dengue/baseline_${SLURM_JOB_ID}.json"

# ── Stage 2: MOG-DFM guided generation ───────────────────────────────────────
echo ""
echo "[Stage 2] MOG-DFM guided generation..."

python peptide_optimization/src/mog_dfm_binding.py \
    --tree-json   "$DENGUE_JSON" \
    --num-peptides 50 \
    --length 12 \
    --top-k 20 \
    --device cuda:0 \
    --output "$REPO/results/dengue/mogdfm_${SLURM_JOB_ID}.json"

echo "[Stage 2] Done: results/dengue/mogdfm_${SLURM_JOB_ID}.json"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "Pipeline complete. Top 5 MOG-DFM peptides:"
python3 - <<'PYEOF'
import json, os, glob
files = sorted(glob.glob("results/dengue/mogdfm_*.json"))
if not files:
    print("  No output file found.")
else:
    data = json.load(open(files[-1]))
    for i, r in enumerate(data[:5], 1):
        print(f"  {i}. {r['sequence']:15s}  weighted_binding={r['weighted_binding']:.4f}")
PYEOF
echo "=============================="
