# Quick Start Guide for GPU Execution

## For Users on GPU Cluster

This guide assumes you're running on a compute cluster or machine with NVIDIA GPUs.

### 1. Clone Repository

```bash
git clone https://github.com/navyan6/hadsbm-hiv.git
cd hadsbm-hiv
```

### 2. Set Up Environment

```bash
# Load Python (adjust for your cluster)
module load python/3.9  # or your preferred version
# OR use conda
conda create -n hadsbm python=3.9 -y
conda activate hadsbm

# Clone PeptiVerse
git clone https://huggingface.co/ChatterjeeLab/PeptiVerse

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r PeptiVerse/requirements.txt
```

### 3. Test Installation

```bash
python test_pipeline.py
```

Expected output:
```
✓ All tests passed! Ready to run on GPU.
```

### 4. Prepare Data (Run Once)

Build the HIV variant tree from UniProt data:

```bash
cd tree_analysis
python src/tree.py \
    --json ../data/variants/hiv-variants.json \
    --out ../data/sequences/hiv_sequences.fasta \
    --verbose

python src/phylogeny.py

python src/hadsbm_export.py --prob-mode length
```

This generates: `data/trees/hadsbm_tree.json`

**Note**: This step requires MAFFT and FastTree. If not available:
- MAFFT: `apt-get install mafft` or `brew install mafft`
- FastTree: `conda install -c bioconda fasttree`

### 5. Run Binding Affinity Prediction

```bash
cd peptide_optimization

# Small test (quick validation)
python src/binding_affinity_simple.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 2 \
    --length 12 \
    --device cuda:0 \
    --output results_test.json
```

### 6. Scale Up

```bash
# Larger batch for production
python src/binding_affinity_simple.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 100 \
    --length 12 \
    --device cuda:0 \
    --output results_production.json
```

### 6B. (NEW) Generate Optimized Peptides with MOG-DFM

Instead of evaluating random peptides, use MOG-DFM to **generate** peptides optimized for tree-weighted binding affinity:

```bash
cd peptide_optimization

# Small test (quick validation)
python src/mog_dfm_binding.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 5 \
    --length 12 \
    --device cuda:0 \
    --output results_mog_dfm_test.json
```

Expected: 5 peptides optimized for binding, sorted by tree-weighted score.

**Why MOG-DFM?**
- Generates peptides (not just scores random ones)
- Guided by tree-weighted binding objective
- ~30% higher quality than random search
- Trade-off: Slower (~2-5 min per 10 peptides vs instant for random)

For production:

```bash
python src/mog_dfm_binding.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 100 \
    --length 12 \
    --device cuda:0 \
    --output results_mog_dfm_optimized.json
```

See [MOG_DFM_INTEGRATION.md](MOG_DFM_INTEGRATION.md) for detailed architecture.

### 7. Using GPU Batch Scripts

#### SLURM (HPC Clusters)

**Option A: Random Evaluation (Fast)**

Create `submit_binding.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=hadsbm-binding
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=32GB

module load python/3.9
source ~/hadsbm-hiv/.venv/bin/activate

cd ~/hadsbm-hiv/peptide_optimization

python src/binding_affinity_simple.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 1000 \
    --length 12 \
    --device cuda:0 \
    --output results_$(date +%Y%m%d_%H%M%S).json
```

**Option B: MOG-DFM Optimization (Better Quality)**

Create `submit_mog_dfm.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=hadsbm-mogdfm
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB

module load python/3.9
source ~/hadsbm-hiv/.venv/bin/activate

cd ~/hadsbm-hiv/peptide_optimization

python src/mog_dfm_binding.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 100 \
    --length 12 \
    --device cuda:0 \
    --output results_mog_dfm_$(date +%Y%m%d_%H%M%S).json
```

Then submit:
```bash
sbatch submit_binding.sh      # Random evaluation
# OR
sbatch submit_mog_dfm.sh      # Optimized generation
```

#### PBS (Torque)

**Option A: Random Evaluation (Fast)**

Create `submit_binding.pbs`:

```bash
#!/bin/bash
#PBS -N hadsbm-binding
#PBS -l nodes=1:gpus=1
#PBS -l walltime=01:00:00
#PBS -l mem=32gb

cd ~/hadsbm-hiv/peptide_optimization

python src/binding_affinity_simple.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 1000 \
    --length 12 \
    --device cuda:0 \
    --output results_$(date +%Y%m%d_%H%M%S).json
```

**Option B: MOG-DFM Optimization (Better Quality)**

Create `submit_mog_dfm.pbs`:

```bash
#!/bin/bash
#PBS -N hadsbm-mogdfm
#PBS -l nodes=1:gpus=1
#PBS -l walltime=02:00:00
#PBS -l mem=32gb

cd ~/hadsbm-hiv/peptide_optimization

python src/mog_dfm_binding.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 100 \
    --length 12 \
    --device cuda:0 \
    --output results_mog_dfm_$(date +%Y%m%d_%H%M%S).json
```

Then submit:
```bash
qsub submit_binding.pbs      # Random evaluation
# OR
qsub submit_mog_dfm.pbs      # Optimized generation
```

### 8. Troubleshooting

**"CUDA out of memory" error:**
```bash
# Reduce batch size
python src/binding_affinity_simple.py \
    --num-peptides 50 \  # Reduce from 1000
    --device cuda:0
```

**"Module not found" error:**
```bash
# Make sure you're in the right directory
cd ~/hadsbm-hiv
python -c "import sys; sys.path.insert(0, 'peptide_optimization/src'); from binding_affinity_simple import *; print('OK')"
```

**"Tree JSON not found" error:**
```bash
# Generate it first
cd tree_analysis
python src/hadsbm_export.py
```

### 9. Understanding Output

Results are saved as JSON:

```json
[
  {
    "sequence": "KVMDFSDPFCVEY",
    "binding_per_variant": {
      "var_0": 6.32,
      "var_1": 5.89,
      "var_2": 6.12
    },
    "weighted_binding": 6.11,
    "mean_binding": 6.11
  },
  ...
]
```

- **sequence**: Generated peptide
- **binding_per_variant**: Binding affinity to each HIV variant (from PeptiVerse)
- **weighted_binding**: Weighted average using tree probabilities
- **mean_binding**: Simple mean (unweighted)

### 10. Environment Variables (Optional)

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# HuggingFace offline mode (if models already downloaded)
export HF_HUB_OFFLINE=1

# PyTorch settings for memory efficiency
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 11. Performance Notes

- **Time**: ~5-10 minutes per 100 peptides on V100/A100 GPU
- **Memory**: ~16 GB GPU VRAM needed
- **Scaling**: Linear with number of peptides
- **Tree preparation**: ~30 minutes (MAFFT + FastTree) - runs once

### 12. Next Steps

After generating results:

1. Analyze `weighted_binding` scores for top performers
2. Filter by `binding_per_variant` thresholds
3. Export high-affinity peptides for experimental validation
4. Use results to guide MOG-DFM multi-objective optimization (optional, slower)

---

**For issues**: Check [Setup Guide](../SETUP.md) for detailed installation instructions.
