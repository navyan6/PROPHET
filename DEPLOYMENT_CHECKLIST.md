# Deployment Checklist: Sequential Execution Guide

## For Someone Else to Run This Pipeline

Execute these files/commands **in order** on a GPU cluster or machine with NVIDIA GPUs.

### Step 1: Setup Environment (One-time)

```bash
# 1a. Clone repository
git clone https://github.com/navyan6/hadsbm-hiv.git
cd hadsbm-hiv

# 1b. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 1c. Clone PeptiVerse
git clone https://huggingface.co/ChatterjeeLab/PeptiVerse
```

**Files involved**: `requirements.txt`

---

### Step 2: Build Phylogenetic Tree (One-time)

Build tree from HIV variants. Requires MAFFT and FastTree system tools.

```bash
# Install system tools (if needed)
# conda install -c bioconda mafft fasttree
# OR: brew install mafft fasttree

# Build tree
cd tree_analysis
python src/tree.py \
    --json ../data/variants/hiv-variants.json \
    --out ../data/sequences/hiv_sequences.fasta
python src/phylogeny.py
python src/hadsbm_export.py --prob-mode length
cd ..
```

**Files executed**:
1. `tree_analysis/src/tree.py` - Extract HIV protease sequences from JSON
2. `tree_analysis/src/phylogeny.py` - MAFFT alignment + FastTree inference
3. `tree_analysis/src/hadsbm_export.py` - Export tree with computed probabilities

**Output**: `data/trees/hadsbm_tree.json` (contains variant sequences + tree probabilities)

---

### Step 3: Test Installation (Optional)

Verify everything works without GPU:

```bash
python test_pipeline.py
```

**File executed**: `test_pipeline.py`

**Expected output**:
```
✓ All tests passed! Ready to run on GPU.
```

---

### Step 4: Generate Peptides with MOG-DFM (Main Algorithm)

Generate peptides optimized for tree-weighted binding affinity.

```bash
cd peptide_optimization
python src/mog_dfm_binding.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 100 \
    --length 12 \
    --device cuda:0 \
    --output results.json
```

**File executed**: `peptide_optimization/src/mog_dfm_binding.py`

**What it does**:
1. Loads tree with variant probabilities
2. Initializes PeptiVerse binding predictor
3. Initializes MOG-DFM solver
4. Runs MOG-DFM optimization loop:
   - Generate candidate peptide with MOG-DFM
   - Evaluate tree-weighted binding (PeptiVerse + tree weights)
   - Output scored peptides (sorted by weighted binding)

**Output**: `results.json` with peptides and binding scores

**Related modules used** (no separate execution needed):
- `peptide_optimization/src/tree_utils.py` - Loaded by mog_dfm_binding.py
- `peptide_optimization/src/binding_affinity_simple.py` - Reference implementation

---

## Quick Summary Table

| Step | File(s) | Purpose | Input | Output |
|------|---------|---------|-------|--------|
| 1a | git clone | Clone repo | - | - |
| 1b | `requirements.txt` | Install deps | pip | - |
| 1c | git clone | Clone PeptiVerse | - | PeptiVerse/ |
| 2a | `tree_analysis/src/tree.py` | Extract sequences | hiv-variants.json | hiv_sequences.fasta |
| 2b | `tree_analysis/src/phylogeny.py` | Build tree | hiv_sequences.fasta | hiv_tree.nwk |
| 2c | `tree_analysis/src/hadsbm_export.py` | Add probabilities | hiv_tree.nwk | **hadsbm_tree.json** ✓ |
| 3 | `test_pipeline.py` | Validate setup | - | Test results |
| 4 | `peptide_optimization/src/mog_dfm_binding.py` | **Main algorithm** | hadsbm_tree.json | **results.json** ✓ |

---

## Alternative: Faster Path (Skip Tree Building)

If you already have `hadsbm_tree.json`, just run:

```bash
cd peptide_optimization
python src/mog_dfm_binding.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 100 \
    --device cuda:0 \
    --output results.json
```

---

## GPU Cluster Batch Script

Save as `submit_mogdfm.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=hadsbm-mogdfm
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=32GB

cd hadsbm-hiv/peptide_optimization

python src/mog_dfm_binding.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 100 \
    --device cuda:0 \
    --output results_$(date +%s).json
```

Submit:
```bash
sbatch submit_mogdfm.sh
```

---

## What Each File Does

### `requirements.txt`
- Lists all Python dependencies (PyTorch, PeptiVerse modules, etc.)
- For macOS: notes about libomp requirement

### `tree_analysis/src/tree.py`
- Reads: `data/variants/hiv-variants.json` (HIV variant data)
- Runs: Extracts 99-aa protease window from each variant
- Outputs: `data/sequences/hiv_sequences.fasta`

### `tree_analysis/src/phylogeny.py`
- Runs: MAFFT (multiple sequence alignment)
- Runs: FastTree (phylogenetic inference)
- Outputs: `data/trees/hiv_tree.nwk` (tree with branch lengths)

### `tree_analysis/src/hadsbm_export.py`
- Reads: `hiv_tree.nwk` + variant sequences
- Computes: Split probabilities from FastTree branch lengths
- Outputs: `data/trees/hadsbm_tree.json` (tree + probabilities + sequences)

### `peptide_optimization/src/tree_utils.py`
- Helper module (auto-imported by mog_dfm_binding.py)
- Loads tree JSON and computes leaf probabilities from splits

### `test_pipeline.py`
- Smoke tests import and basic functionality
- No GPU required
- Verifies: PyTorch, Transformers, PeptiVerse, peptide generation

### `peptide_optimization/src/mog_dfm_binding.py`
- **Main algorithm**
- Combines:
  - MOG-DFM solver (generates peptides)
  - PeptiVerse (evaluates binding)
  - Tree probabilities (weights objectives)
- Output: JSON with optimized peptides

---

## Troubleshooting by File

| File | Error | Fix |
|------|-------|-----|
| `tree_analysis/src/tree.py` | "hiv-variants.json not found" | Verify data/variants directory exists |
| `tree_analysis/src/phylogeny.py` | "mafft: command not found" | `conda install -c bioconda mafft` |
| `tree_analysis/src/phylogeny.py` | "FastTree not found" | `conda install -c bioconda fasttree` |
| `tree_analysis/src/hadsbm_export.py` | "tree.nwk not found" | Run phylogeny.py first |
| `test_pipeline.py` | Import errors | Run: `pip install -r requirements.txt` |
| `mog_dfm_binding.py` | "PeptiVerse not found" | `git clone https://huggingface.co/ChatterjeeLab/PeptiVerse` |
| `mog_dfm_binding.py` | CUDA OOM | Reduce `--num-peptides` (50 → 10) |
| `mog_dfm_binding.py` | "hadsbm_tree.json not found" | Run tree building pipeline (Steps 2a-2c) |

---

## Success Indicators

✓ Step 1: `pip install` completes without errors  
✓ Step 2a: `hiv_sequences.fasta` created  
✓ Step 2b: `hiv_tree.nwk` created  
✓ Step 2c: `data/trees/hadsbm_tree.json` created  
✓ Step 3: Test passes with "✓ All tests passed"  
✓ Step 4: `results.json` created with peptides + scores  

---

## Summary

**Minimal execution**:
```bash
# Setup
pip install -r requirements.txt
git clone https://huggingface.co/ChatterjeeLab/PeptiVerse

# Tree (one-time)
cd tree_analysis && python src/hadsbm_export.py && cd ..

# Algorithm (repeat as needed)
cd peptide_optimization
python src/mog_dfm_binding.py --tree-json ../data/trees/hadsbm_tree.json --device cuda:0
```

**Total workflow**: ~45 min (30 min tree building + 15 min MOG-DFM generation for 100 peptides on V100 GPU)
