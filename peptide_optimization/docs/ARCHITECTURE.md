# Generalized Multi-Objective Peptide Optimization

## Framework Stack

This implementation combines:
- **PepDFM**: Discrete flow matching for peptide generation
- **MOG-DFM**: Multi-objective guidance framework  
- **PeptiVerse**: Binding affinity predictor
- **MOG-DFM Classifiers**: Drug-like property evaluation

## Design Principles

### 1. **Multi-Target Abstraction**
The code is designed to handle optimization across **any number of target variants**:

- **Single unified affinity predictor** evaluates all targets
- **Automatic target affinity averaging** for multi-objective guidance
- **Per-target affinity tracking** for result analysis

### 2. **Modular Property Scoring**
Property models are loaded once and reused across all targets:
- One instance of each property predictor (hemolysis, solubility, etc.)
- Vectorized evaluation where possible
- Clear separation of concerns

### 3. **Configuration-Driven Flexibility**
All hyperparameters are configurable via the `Config` dataclass:
- Easy to adjust for different use cases
- Weights can be tuned per objective
- Model paths are customizable

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         optimize_peptides_moo.py                        │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Config (Dataclass)                               │  │
│  │ - peptide_length, num_batches, device            │  │
│  │ - model paths, objective_weights                 │  │
│  └──────────────────────────────────────────────────┘  │
│                      ↓                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ TargetVariant(s)                                 │  │
│  │ - name, sequence, input_ids (tokenized)          │  │
│  └──────────────────────────────────────────────────┘  │
│                      ↓                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ PeptideOptimizer                                 │  │
│  │                                                   │  │
│  │ ┌────────────────────────────────────────────┐  │  │
│  │ │ solver (PepDFM)                             │  │  │
│  │ │ - Sequence generation via discrete DFM     │  │  │
│  │ └────────────────────────────────────────────┘  │  │
│  │                                                   │  │
│  │ ┌────────────────────────────────────────────┐  │  │
│  │ │ PropertyScorer                              │  │  │
│  │ │                                              │  │  │
│  │ │ ├─ affinity_models (PeptiVerse WT model)   │  │  │
│  │ │ ├─ hemolysis_model (MOG-DFM)               │  │  │
│  │ │ ├─ nonfouling_model (MOG-DFM)              │  │  │
│  │ │ ├─ solubility_model (MOG-DFM)              │  │  │
│  │ │ └─ halflife_model (MOG-DFM)                │  │  │
│  │ └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│                      ↓                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ MOG-DFM Guided Optimization Loop                │  │
│  │                                                   │  │
│  │ For each batch:                                 │  │
│  │  1. Generate random peptide (PepDFM base)      │  │
│  │  2. MOG-DFM multi-objective guidance           │  │
│  │  3. Evaluate all objectives (PeptiVerse + MOG) │  │
│  │  4. Store PeptideScores result                 │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Handling Multiple Variants

### Current Implementation
```python
# Load 3 HIV variants
targets = [
    TargetVariant("HXB2", sequence_hxb2, device, tokenizer),
    TargetVariant("NL4.3", sequence_nl43, device, tokenizer),
    TargetVariant("JRFL", sequence_jrfl, device, tokenizer),
]

optimizer = PeptideOptimizer(config, targets)
results = optimizer.optimize()

# Each peptide is evaluated against all 3 targets
# Example result:
# {
#   "sequence": "MVKKGLPKEYPRQ",
#   "affinity_per_target": {
#     "HXB2": 6.5,
#     "NL4.3": 6.2,
#     "JRFL": 5.9
#   },
#   ...
# }
```

### Key Features

**1. PeptiVerse Binding Affinity Predictor**
- Uses PeptiVerse's trained `best_model_wt` (WT/peptide binding)
- One model evaluates all targets
- Reused across all batch iterations
- Scales linearly with number of targets
- Provides robust, cross-validated binding predictions

**2. Automatic Objective Aggregation**
In `PropertyScorer.score_properties()`:
```python
# Compute affinity for each target
for target_name, affinity_model in affinity_models.items():
    affinity = affinity_model(peptide_ids)
    target_affinities[target_name] = affinity.item()

# Average across targets for MOG-DFM guidance
avg_affinity = np.mean(list(target_affinities.values()))
```

**3. Per-Target Tracking**
Results include `affinity_per_target` dict for post-hoc analysis:
- Identify peptides that bind to specific variants
- Detect sequence-specific selectivity
- Cross-variant comparison

