# Setup Guide for hadsbm-hiv Repository

## Prerequisites

- Python 3.9+
- CUDA 12.4 (or compatible PyTorch version)
- `pip` and `conda` (recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/navyan6/hadsbm-hiv.git
cd hadsbm-hiv
```

### 2. Set Up Python Environment

```bash
# Create conda environment
conda create -n hadsbm python=3.9 -y
conda activate hadsbm

# Or with venv
python3.9 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up MOG-DFM

MOG-DFM is included in the repository. Install its requirements:

```bash
pip install -r MOG-DFM/requirements.txt
```

**Note**: MOG-DFM requires specific versions of:
- PyTorch 2.4.0 with CUDA 12.4
- transformers (for model loading)
- xgboost (for classifiers)

If you're on a different CUDA version, update PyTorch:
```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. Set Up PeptiVerse

PeptiVerse is available on HuggingFace and needs to be cloned separately:

```bash
# Clone PeptiVerse into the repository root
git clone https://huggingface.co/ChatterjeeLab/PeptiVerse
cd PeptiVerse
pip install -r requirements.txt
cd ..
```

**Directory structure should look like:**
```
hadsbm-hiv/
├── MOG-DFM/          (included in repo)
├── PeptiVerse/       (cloned from HuggingFace)
├── peptide_optimization/
├── tree_analysis/
├── data/
└── requirements.txt
```

### 6. Download Pre-trained Models (if needed)

Some models are automatically downloaded on first use via HuggingFace. To pre-download:

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')"
```

### 7. Verify Installation

```bash
# Test imports
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'MOG-DFM'))
sys.path.insert(0, str(Path.cwd() / 'PeptiVerse'))
from models.peptide_classifiers import HemolysisModel
from inference import PropertyPredictor
print('✓ All imports successful')
"

# Run a simple example
cd peptide_optimization
python examples/examples_optimize.py  # Will demonstrate all 4 examples
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'models'`

**Problem**: MOG-DFM path not in sys.path

**Solution**:
```bash
# Make sure you're in the repository root
cd hadsbm-hiv
python -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path.cwd() / 'MOG-DFM')); from models.peptide_classifiers import HemolysisModel; print('OK')"
```

### `ModuleNotFoundError: No module named 'inference'`

**Problem**: PeptiVerse not cloned or not in path

**Solution**:
```bash
# Verify PeptiVerse directory exists
ls PeptiVerse/inference.py

# If missing, clone it
git clone https://huggingface.co/ChatterjeeLab/PeptiVerse
```

### `CUDA out of memory` when running examples

**Problem**: GPU memory exhausted

**Solution**:
```python
# Reduce batch size in Config
from optimize_peptides_moo import Config
config = Config()
config.num_batches = 1  # Reduced from 2
config.num_samples = 1  # Reduced from default
```

### `ImportError: cannot import name 'PropertyPredictor'` 

**Problem**: PeptiVerse version mismatch or inference.py not at root

**Solution**:
```bash
# Check PeptiVerse structure
ls -la PeptiVerse/
# Should show: inference.py, __init__.py, classifier_code/, models/, etc.

# If inference.py is in a subdirectory, update the import path in optimize_peptides_moo.py
```

## Running the Peptide Optimization Pipeline

### Example 1: Single Target (HIV WT)
```bash
cd peptide_optimization
python -c "from src.optimize_peptides_moo import main; results = main(output_file='results.json'); print(f'Generated {len(results)} sequences')"
```

### Example 2: Multiple Targets
```python
from src.optimize_peptides_moo import main
results = main(targets=["MVKKGL...", "MVKKGM...", "MVKKGN..."], output_file="results_multi.json")
```

### Example 3: Custom Configuration
```python
from src.optimize_peptides_moo import Config, PeptideOptimizer, TargetVariant
config = Config(peptide_length=15, num_batches=4)
# ... rest of setup
optimizer = PeptideOptimizer(config, targets)
results = optimizer.optimize()
```

## Running the Tree Analysis Pipeline

### Full Pipeline
```bash
cd tree_analysis
python pipelines/pipeline_basic.py \
  --skip-fasta \
  --prob-mode length \
  --ascii-tree
```

### Individual Steps
```bash
# 1. Generate variants
python src/tree.py --json ../data/variants/hiv-variants.json --out ../data/sequences/hiv_sequences.fasta

# 2. Tree inference
python src/phylogeny.py

# 3. Visualize
python src/visualize_tree.py --ascii

# 4. Export HadSBM JSON
python src/hadsbm_export.py --prob-mode length
```

## Development Setup

If you're modifying the code and want to use editable installs:

```bash
# Install in development mode
pip install -e .

# Or for specific modules
cd MOG-DFM
pip install -e .
cd ../PeptiVerse
pip install -e .
```

## Environment Variables (Optional)

```bash
# Set device explicitly
export TORCH_DEVICE=cuda:0

# Disable pre-training download
export HF_HUB_OFFLINE=0
```

## Next Steps

1. Read [peptide_optimization/docs/ARCHITECTURE.md](../peptide_optimization/docs/ARCHITECTURE.md) for technical details
2. Check [tree_analysis/docs/PIPELINE_EXPLANATION.md](../tree_analysis/docs/PIPELINE_EXPLANATION.md) for tree building
3. Run examples in `peptide_optimization/examples/`
4. Review data structure in `data/` folder

## Support

For issues:
1. Check that MOG-DFM and PeptiVerse directories exist
2. Verify all packages in requirements.txt are installed
3. Ensure Python 3.9+ and CUDA 12.4 compatibility
4. See specific module READMEs for detailed configuration options
