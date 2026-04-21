#!/bin/bash
# Run this ONCE on the GT ICE login node to set up the environment.
# Usage: bash setup_cluster_env.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Setting up environment in: $REPO_DIR"

# Load conda module (GT ICE)
module load anaconda3

# Create env if it doesn't exist
if conda info --envs | grep -q "hadsbm"; then
    echo "[INFO] Conda env 'hadsbm' already exists"
else
    echo "[INFO] Creating conda env 'hadsbm' (Python 3.9)..."
    conda create -n hadsbm python=3.9 -y
fi

conda activate hadsbm

# Install MAFFT + FastTree (needed for tree building)
conda install -y -c bioconda mafft fasttree

# Core pip dependencies
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.18.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install -r "$REPO_DIR/requirements.txt"

# PeptiVerse dependencies
if [ -f "$REPO_DIR/PeptiVerse/requirements.txt" ]; then
    pip install -r "$REPO_DIR/PeptiVerse/requirements.txt"
fi

# Pre-download ESM2 tokenizer (avoids network access on compute nodes)
python -c "
from transformers import AutoTokenizer
print('Downloading ESM2 tokenizer...')
AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
print('Done.')
"

echo ""
echo "=============================="
echo "Environment setup complete."
echo "Activate with: conda activate hadsbm"
echo "Submit job with: sbatch run_dengue_pipeline.slurm"
echo "=============================="
