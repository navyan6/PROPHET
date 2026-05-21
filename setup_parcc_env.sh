#!/bin/bash
# Run ONCE on the PARCC login node to set up the Python environment.
# Usage: bash setup_parcc_env.sh
#
# Fill in your HF token below if you have one (speeds up model downloads
# and avoids rate limits on shared nodes). Get one at:
# https://huggingface.co/settings/tokens  (read-only token is fine)
HF_TOKEN=""   # ← optional but recommended

set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Repo: $REPO_DIR"

# ── 0. Redirect HuggingFace cache to scratch (avoids filling home quota) ──────
# ESM2-650M alone is ~2.5 GB — do not let it land in $HOME/.cache
# PARCC uses VAST storage — no $SCRATCH; use project dir instead
export HF_HOME="/vast/projects/pranam/lab/$USER/.cache/huggingface"
mkdir -p "$HF_HOME"
echo "[INFO] HuggingFace cache → $HF_HOME"

# ── 1. Load conda ─────────────────────────────────────────────────────────────
module load miniconda3/25.5.1
source activate base 2>/dev/null || true

# ── 2. Create env ─────────────────────────────────────────────────────────────
if conda info --envs | grep -q "hadsbm"; then
    echo "[INFO] Conda env 'hadsbm' already exists — skipping creation"
else
    echo "[INFO] Creating conda env 'hadsbm' (Python 3.10)..."
    conda create -n hadsbm python=3.10 -y
fi
source activate hadsbm

# ── 3. PyTorch with CUDA ──────────────────────────────────────────────────────
# PARCC has B200 (Blackwell) GPUs — requires PyTorch 2.7+ with CUDA 12.8
pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── 4. Core dependencies ──────────────────────────────────────────────────────
pip install -r "$REPO_DIR/requirements.txt"
pip install -r "$REPO_DIR/PeptiVerse/requirements.txt"

# flow-matching is needed by MOG-DFM but not in requirements.txt
pip install flow-matching==1.0.10

# ── 5. Bioinformatics tools ───────────────────────────────────────────────────
conda install -y -c bioconda mafft fasttree

# ── 6. Pre-download HuggingFace models on the login node ─────────────────────
# Compute nodes on PARCC may not have outbound internet — download here first.
echo ""
echo "[INFO] Pre-downloading HuggingFace models (this may take a few minutes)..."
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
fi

python - <<'PYEOF'
from transformers import EsmModel, EsmTokenizer, AutoTokenizer
import sys

print("  Downloading ESM2-650M (for PeptiVerse)...")
EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", add_pooling_layer=False)
print("  Done: ESM2-650M")

print("  Downloading ESM2-650M tokenizer (for MOG-DFM)...")
AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
print("  Done: tokenizer")

print("  All HuggingFace models cached.")
PYEOF

# ── 7. Smoke-test the full import chain ───────────────────────────────────────
echo ""
echo "[INFO] Testing import chain..."
python - <<'PYEOF'
import sys, os
repo = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
sys.path.insert(0, os.path.join(repo, "MOG-DFM"))

from prophet.stage2 import AffinityScorer
from models.peptide_classifiers import load_solver
print("  Imports OK")
PYEOF

echo ""
echo "=============================="
echo "Setup complete."
echo "Activate with: source activate hadsbm"
echo "Submit job with: sbatch run_hiv_stage2_parcc.slurm"
echo "=============================="
