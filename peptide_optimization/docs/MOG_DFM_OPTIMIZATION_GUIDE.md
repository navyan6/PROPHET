# Multi-Objective Peptide Optimization with MOG-DFM

This guide covers using `optimize_peptides_moo.py` for generating peptides optimized across multiple target variants and biophysical properties.

## Overview

The script performs multi-objective guided optimization to generate peptides with:

- **Primary Objective**: Binding affinity to multiple target protein variants
- **Secondary Objectives**:
  - Hemolysis (toxicity) - should be low
  - Non-fouling (biofouling resistance) - should be high
  - Solubility - should be high
  - Half-life - should be long/high

## Quick Start

### Basic Usage (Single Target - HIV WT)

```python
from optimize_peptides_moo import main

# Uses default HIV WT from wildtype.fasta
results = main()
```

### Multi-Target Optimization

```python
from optimize_peptides_moo import main

# Optimize across multiple HIV variants
variants = [
    "MGARASVLSGGELDRWEKIRLRPGGKKKYKLKHIVWASRELERFAVNPGLLETSEGCRQI...",
    "MGARASVLSGGELDRWEKIRLRPGGKKKYKLKHIVWASRELERFAVNPGLLETSEGCRQV...",  
]

results = main(targets=variants, output_file="multi_target_results.json")
```

### Custom Configuration

```python
from optimize_peptides_moo import main, Config

# Create custom configuration for longer peptides,more optimization
config = Config()
config.peptide_length = 15           
config.num_batches = 5               
config.num_steps = 200   
#manually enter the weights for objective importance, weight             
config.objective_weights = {
    "affinity": 2.0,      
    "hemolysis": 1.0,
    "nonfouling": 1.0,
    "solubility": 0.5,
    "halflife": 0.5,
}

# Note: Would need to modify main() to accept config, or use:
from optimize_peptides_moo import PeptideOptimizer, TargetVariant
from pathlib import Path
from transformers import AutoTokenizer

targets = [...]
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
target_variants = [
    TargetVariant(f"target_{i}", seq, config.device, tokenizer)
    for i, seq in enumerate(targets)
]
optimizer = PeptideOptimizer(config, target_variants)
results = optimizer.optimize()
```

## Architecture

### Key Classes

#### `Config`
Configuration dataclass with all hyperparameters:
- Model paths (solver, classifiers)
- Optimization parameters (length, batches, steps)
- Device (GPU/CPU)
- Objective weights

#### `TargetVariant`
Represents a target protein/variant:
- Name and sequence
- Pre-tokenized for efficient model loading

#### `PropertyScorer`
Loads and manages all property prediction models:
- Affinity predictors (one per target)
- Hemolysis, non-fouling, solubility, half-life models
- Provides unified scoring interface

#### `PeptideOptimizer`
Main pipeline:
- Initializes solver and scorers
- Generates random starting peptides
- Runs multi-objective guided sampling
- Scores and decodes optimized peptides

### Data Flow

```
Targets (FASTA or sequences)

TargetVariant objects

PeptideOptimizer initialization
    ├─ Load DFM solver
    └─ Load PropertyScorer (affinity + biophysical models)

For each batch:
    1. Generate random peptide
    2. Multi-objective guided sampling
    3. Score all properties for all targets
    4. Store results

JSON output with trajectories and scores
```

## Output Format

Results are saved as JSON with structure:

```json
[
  {
    "sequence": "MVKKGLPKEYPRQ",
    "affinity_per_target": {
      "target_0": 6.5,
      "target_1": 6.2
    },
    "hemolysis": 0.85,
    "nonfouling": 0.72,
    "solubility": 0.91,
    "halflife": 1.8
  },
  ...
]
```

## Scaling to Many Variants

The current implementation:
- Loads one affinity predictor shared across all targets
- Computes affinity separately for each target variant
- Averages affinity across targets as the optimization objective

To optimize against many variants efficiently:

1. **Batch Processing**: Process variants in groups
2. **Target-Specific Models**: Extend `PropertyScorer.load_affinity_predictors()` to load variant-specific models if available
3. **Filtering**: Pre-filter germane variants, optimize subset


