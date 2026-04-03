# hadsbm-hiv: HIV Variant Binding Affinity Prediction

Fast, GPU-ready pipeline to compute binding affinity of peptides to HIV protease variants using tree-weighted probabilities.

## 🎯 What It Does

1. **Loads HIV variants** from phylogenetic tree analysis (with variant probabilities)
2. **Generates random peptides** 
3. **Computes binding affinity** to each variant using PeptiVerse (state-of-the-art ML model)
4. **Weights by tree probabilities** to get variant-specific predictions
5. **Outputs peptide rankings** by weighted binding affinity

## 🚀 Quick Start

### For GPU Clusters ⭐ **START HERE**

👉 [**GPU_QUICK_START.md**](GPU_QUICK_START.md) - Complete guide with SLURM/PBS batch scripts

### For Local Development (CPU)

```bash
# Clone and setup
git clone https://github.com/navyan6/hadsbm-hiv.git
cd hadsbm-hiv

# Install dependencies
pip install -r requirements.txt
git clone https://huggingface.co/ChatterjeeLab/PeptiVerse

# Test installation
python test_pipeline.py

# Run demo (CPU mode)
make binding
```

## 📋 Requirements

- **Python** 3.9+
- **GPU** (optional): NVIDIA GPU for fast computation
- **CPU only**: Works but slower (~1 min per 10 peptides)

## 📂 Structure

```
hadsbm-hiv/
├── peptide_optimization/src/
│   ├── binding_affinity_simple.py     # Main pipeline
│   └── tree_utils.py                  # Tree data loading
├── tree_analysis/src/                 # Phylogenetic tree building
│   ├── tree.py
│   ├── phylogeny.py
│   └── hadsbm_export.py
├── MOG-DFM/                           # Multi-objective generation (optional)
├── PeptiVerse/                        # Binding affinity models
├── data/                              # HIV variants and sequences
└── GPU_QUICK_START.md                 # 👈 GPU users: read this!
```

## 🔧 Usage

### Basic (Default HIV WT + Variants)

```bash
cd peptide_optimization
python src/binding_affinity_simple.py --num-peptides 10 --device cpu
```

### With Tree Data (Production)

```bash
# 1. Build tree (one-time)
make tree

# 2. Run peptide evaluation on GPU
cd peptide_optimization
python src/binding_affinity_simple.py \
    --tree-json ../data/trees/hadsbm_tree.json \
    --num-peptides 1000 \
    --length 12 \
    --device cuda:0 \
    --output results.json
```

## 📊 Output Example

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
  }
]
```

- **weighted_binding**: Tree-weighted average (uses variant probabilities)
- **binding_per_variant**: Individual binding scores from PeptiVerse
- **mean_binding**: Simple unweighted average

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Run `python test_pipeline.py` |
| CUDA out of memory | Reduce `--num-peptides` or use `--device cpu` |
| PeptiVerse not found | `git clone https://huggingface.co/ChatterjeeLab/PeptiVerse` |
| Tree JSON missing | Run `make tree` (requires MAFFT + FastTree) |

## 📚 Documentation

- [GPU_QUICK_START.md](GPU_QUICK_START.md) - Cluster execution guide
- [SETUP.md](SETUP.md) - Detailed installation
- [tree_analysis/docs/PIPELINE_EXPLANATION.md](tree_analysis/docs/PIPELINE_EXPLANATION.md) - Tree building details

## 🔬 Models

- **Binding Affinity**: PeptiVerse (`best_model_wt`) - WT/peptide binding prediction
- **Tree Structure**: FastTree maximum-likelihood phylogeny
- **Sequence Alignment**: MAFFT

## 📝 Citation

If used in research, please cite:
- PeptiVerse: https://huggingface.co/ChatterjeeLab/PeptiVerse
- FastTree: Price et al. (2010)
- MAFFT: Katoh & Standley (2013)

## 🤝 License

See LICENSE file.

