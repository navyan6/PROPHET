# Quick Reference: Multi-Objective Peptide Optimization

## Files Overview

| File | Purpose |
|------|---------|
| `optimize_peptides_moo.py` | Main optimization script |
| `examples_optimize.py` | 4 runnable examples demonstrating different use cases |
| `MOG_DFM_OPTIMIZATION_GUIDE.md` | User guide with examples |
| `ARCHITECTURE.md` | Technical design documentation |

## Common Tasks

### Task 1: Optimize Against Single Target (HIV WT)
```python
from optimize_peptides_moo import main

# Uses wildtype.fasta by default
results = main()
```

### Task 2: Optimize Against Multiple Variants
```python
from optimize_peptides_moo import main, read_fasta

# Load variants from FASTA
variants_dict = read_fasta("my_variants.fasta")
sequences = list(variants_dict.values())

results = main(targets=sequences, output_file="multi_var_results.json")
```

### Task 3: Adjust Optimization Parameters
```python
from optimize_peptides_moo import Config, PeptideOptimizer, TargetVariant
from transformers import AutoTokenizer

# Create custom config, adapt to longer peptides/more iteration
config = Config()
config.peptide_length = 15           
config.num_batches = 5         
config.objective_weights = {
    "affinity": 2.0,    
    "hemolysis": 1.0,
    "nonfouling": 1.0,
    "solubility": 0.5,
    "halflife": 0.5,
}

# Set up targets/esm
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
targets = [TargetVariant("target", seq, config.device, tokenizer) for seq in sequences]

# Run optimization
optimizer = PeptideOptimizer(config, targets)
results = optimizer.optimize()
```

### Task 4: Analyze Results
```python
import json

with open("results.json") as f:
    results = json.load(f)

# Summary statistics
affinities = [max(r['affinity_per_target'].values()) for r in results]
print(f"Mean affinity: {sum(affinities)/len(affinities):.3f}")
print(f"Best affinity: {max(affinities):.3f}")

# Find best binder
best = max(results, key=lambda r: max(r['affinity_per_target'].values()))
print(f"Best sequence: {best['sequence']}")

# Filter by criteria
good_binders = [r for r in results if max(r['affinity_per_target'].values()) > 6.0]
non_hemolytic = [r for r in good_binders if r['hemolysis'] > 0.8]
```

### Task 5: Batch Process Large Variant Sets
```python
from optimize_peptides_moo import main, read_fasta

variants = read_fasta("100_variants.fasta")
sequences = list(variants.values())

# Process in batches of 20
batch_size = 20
all_results = []

for i in range(0, len(sequences), batch_size):
    batch = sequences[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1}...")
    
    results = main(
        targets=batch,
        output_file=f"batch_{i//batch_size}.json"
    )
    all_results.extend(results)

print(f"Total: {len(all_results)} peptides optimized")

# Analyze across all batches
import json
with open("all_results.json", 'w') as f:
    json.dump(all_results, f)
```

## Configuration Cheat Sheet

```python
config = Config()

# Peptide properties
config.peptide_length = 12          
config.num_samples = 1              
config.num_batches = 2              

# Sampling
config.num_steps = 100              
config.step_size = 1/200            

# Objectives (example weights)
config.objective_weights = {
    "affinity": 2.0,        
    "hemolysis": 1.0,       
    "nonfouling": 0.5,     
    "solubility": 1.0,
    "halflife": 0.5,
}

# Hardware
config.device = "cuda:0"            
```

## Performance Benchmarks

| Task | Time | GPU Memory | Notes |
|------|------|-----------|-------|
| Single batch (12 aa) | 30-60s | 8-10 GB | RTX 3090, CUDA 12.4 |
| 10 batches | 5-10 min | 8-10 GB | Minimal memory overhead |
| 20 targets | +linear | +minimal | Affinity scoring is main cost |
| CPU mode | 5-10x slower | - | Not recommended for production |

## Troubleshooting

### Issue: CUDA out of memory
```python
# Solution: Reduce peptide length or increase batch wait time
config.peptide_length = 10
# Or use CPU (slow)
config.device = "cpu"
```

### Issue: Model checkpoint not found
```python
# Verify MOG-DFM is properly cloned
import os
print(os.listdir("MOG-DFM/classifier_ckpt/"))  # Should see .pt and .json files

# Verify paths
from pathlib import Path
print(Path(__file__).parent / "MOG-DFM")
```

### Issue: Poor peptide diversity
```python
# Increase number of batches and steps
config.num_batches = 10
config.num_steps = 200

# Reduce affinity weight to explore more
config.objective_weights["affinity"] = 1.0  # Was 2.0
config.objective_weights["solubility"] = 1.5  # Emphasize other properties
```

## Output Interpretation

```json
{
  "sequence": "MVKKGLPKEYPRQ",
  "affinity_per_target": {
    "target_0": 6.5,      // Higher = better binding (log pKd)
    "target_1": 6.2
  },
  "hemolysis": 0.85,      // Higher = less toxic (0-1 scale)
  "nonfouling": 0.72,     // Higher = more biofouling resistant (0-1)
  "solubility": 0.91,     // Higher = more soluble (0-1)
  "halflife": 1.8         // Higher = longer half-life (log hours)
}
```

Enable verbose output:
```python
optimizer = PeptideOptimizer(config, targets)
results = optimizer.optimize()  # Prints progress + scores for each batch
```

Check intermediate values:
```python
peptide_ids = optimizer.generate_random_peptide(config.peptide_length)
scores, affinities = optimizer.scorer.score_properties(peptide_ids, optimizer.affinity_models)
print(f"Raw scores: {scores}")
print(f"Per-target affinities: {affinities}")
```
