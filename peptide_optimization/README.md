# Peptide Optimization

Multi-objective optimization of peptide sequences using **PepDFM + MOG-DFM** framework with **PeptiVerse** property prediction.

## Structure

- **`src/`** - Source code
  - `optimize_peptides_moo.py` - Main optimization pipeline

- **`examples/`** - Example usage scripts
  - `examples_optimize.py` - 4 working examples

- **`docs/`** - Documentation
  - `ARCHITECTURE.md` - Technical design
  - `QUICK_REFERENCE.md` - Cheat sheet
  - `GUIDE.md` - User guide

- **`validation/`** - Validation & testing
  - `VALIDATION_CHECKLIST.md` - Feature verification

## Integration

### Generative Model
- **PepDFM**: Unconditional discrete flow matching for peptide generation
- **MOG-DFM**: Multi-objective guidance framework

### Property Prediction
- **Binding Affinity**: PeptiVerse `best_model_wt` (WT/peptide binding)
- **Hemolysis**: MOG-DFM trained classifier
- **Non-fouling**: MOG-DFM trained classifier  
- **Solubility**: MOG-DFM trained classifier
- **Half-life**: MOG-DFM trained transformer model

## Key Features

### Multi-Target Support
```python
# Optimize across ANY number of variants - no code duplication
results = main(targets=[seq1, seq2, seq3, ...])

# Each result includes per-target affinity via PeptiVerse:
{
  "sequence": "MVKKGLPKEYPRQ",
  "affinity_per_target": {
    "target_0": 6.5,
    "target_1": 6.2,
    "target_2": 5.9
  },
  ...
}
```

### Property Scoring
- Clean integration of PeptiVerse binding affinity
- MOG-DFM drug-like property predictors
- Configurable multi-objective weights
- Per-target tracking and analysis

## Quick Start

### Basic Usage (Single Target - HIV WT)

```python
from src.optimize_peptides_moo import main

results = main()
```

### Multi-Target Optimization

```python
from src.optimize_peptides_moo import main

variants = [seq1, seq2, seq3]
results = main(targets=variants, output_file="results.json")
```

### Custom Configuration

```python
from src.optimize_peptides_moo import Config, PeptideOptimizer, TargetVariant
from transformers import AutoTokenizer

config = Config()
config.peptide_length = 15
config.objective_weights = {
    "affinity": 2.0,
    "hemolysis": 1.0,
    ...
}

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
targets = [TargetVariant(name, seq, config.device, tokenizer) for ...]
optimizer = PeptideOptimizer(config, targets)
results = optimizer.optimize()
```

## Requirements

- MOG-DFM (parent directory)
- PeptiVerse (parent directory)
- PyTorch with CUDA
- transformers, xgboost
- See MOG-DFM and PeptiVerse requirements

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md)
- [Quick Reference](docs/QUICK_REFERENCE.md)
- [User Guide](docs/GUIDE.md)
- [Validation Checklist](validation/VALIDATION_CHECKLIST.md)

## Examples

Run the example scripts:
```bash
cd examples
python examples_optimize.py 1  # Basic optimization
python examples_optimize.py 2  # Multi-target
python examples_optimize.py 3  # Custom config
python examples_optimize.py 4  # Analyze results
```
