# Tree-Derived Probability Weighting for MOG-DFM

## Overview

The binding affinity pipeline now uses **tree-derived probabilities** as the weight vector for MOG-DFM optimization. Each HIV variant's probability is computed from the phylogenetic tree structure, not assumed to be uniform.

## How Probabilities Are Computed

Each leaf's probability is the **product of split probabilities** along the path from tree root to that leaf:

### Tree Structure (from `hadsbm_tree.json`)

```json
{
  "splits": [
    {
      "parent_index": 0,
      "left_child_index": 1,
      "right_child_index": 2,
      "p_left": 0.6,           // Split probability: left branch
      "p_right": 0.4           // Split probability: right branch
    },
    {
      "parent_index": 1,
      "left_child_index": 3,
      "right_child_index": 4,
      "p_left": 0.5,
      "p_right": 0.5
    }
  ],
  "leaf_endpoints_pi": [
    {"node_index": 3, "leaf_id": "var_0", "sequence": "..."},
    {"node_index": 4, "leaf_id": "var_1", "sequence": "..."},
    {"node_index": 2, "leaf_id": "var_2", "sequence": "..."}
  ]
}
```

### Computation Example

For a tree with 3 leaves:

```
     Root (0)
    /  (0.6)  \  (0.4)
   N1          Leaf2
  / (0.5) \ (0.5)
Leaf0    Leaf1

Leaf0 probability = 0.6 × 0.5 = 0.30
Leaf1 probability = 0.6 × 0.5 = 0.30
Leaf2 probability = 0.4       = 0.40
Sum             = 1.00 ✓
```

## Implementation

### `tree_utils.py`

**Key functions:**
- `_build_tree_structure()` - Convert splits array into traversable format
- `_find_parent()` - Find parent node of any child
- `_compute_leaf_probability()` - Trace path to leaf and multiply probabilities
- `load_tree_probabilities()` - Load all variants with computed probabilities

**Uses:**
```python
from peptide_optimization.src.tree_utils import load_tree_probabilities

variants = load_tree_probabilities("data/trees/hadsbm_tree.json")
# Returns: List[VariantWithProbability(name, sequence, probability)]
```

### `binding_affinity_simple.py`

**Usage in weight vector:**
```python
# Weighted binding affinity (for MOG-DFM objective)
weighted_binding = sum(
    binding_score * variant.probability
    for binding_score, variant in zip(scores, variants)
)
```

This produces a vector of probabilities that MOG-DFM can use to weight multiple objectives across variants.

## Output

Each peptide result includes:

```json
{
  "sequence": "KVMDFSDPFCVEY",
  "binding_per_variant": {
    "var_0": 6.32,
    "var_1": 5.89,
    "var_2": 6.12
  },
  "weighted_binding": 6.11,     // Σ(score_i × probability_i)
  "mean_binding": 6.11          // Simple average
}
```

- **weighted_binding**: Tree-weighted score used for MOG-DFM optimization
- **binding_per_variant**: Individual scores before weighting
- **mean_binding**: Unweighted average for comparison

## Testing

Test validates tree computation with a known tree:

```bash
python test_pipeline.py
# Output:
# ✓ Tree utilities
# ✓ Probabilities sum to 1.0
# ✓ leaf_0: 0.3000 (expected 0.3000)
# ✓ leaf_1: 0.3000 (expected 0.3000)
# ✓ leaf_2: 0.4000 (expected 0.4000)
```

## Integration with Tree Pipeline

The tree JSON comes from `tree_analysis/src/hadsbm_export.py`:

```bash
# Build phylogenetic tree (includes probability computation)
cd tree_analysis
python src/hadsbm_export.py \
  --nwk hiv_tree.nwk \
  --fasta hiv_sequences_aligned.fasta \
  --variants-json hiv-variants.json \
  --prob-mode length  # or 'uniform'
```

Output: `data/trees/hadsbm_tree.json` with computed split probabilities.

## Probability Modes

The tree pipeline supports different probability modes (set in hadsbm_export.py):

- **`length`** (default): `p_left = left_branch_length / total_branch_length`
  - Longer branches = higher probability
  - Reflects evolutionary distance
  
- **`uniform`**: `p_left = 0.5, p_right = 0.5`
  - Equal probability at each split
  - Simple baseline

## GPU Usage (MOG-DFM)

The probability weight vector is passed to the MOG-DFM multi-objective optimizer:

```python
# Conceptual (full MOG-DFM would do this)
objectives = {
    "binding": [weighted_binding_var_0, weighted_binding_var_1, ...]
    # Other objectives: hemolysis, solubility, etc.
}

# MOG-DFM optimizes with weights = [0.30, 0.30, 0.40]
optimized_peptides = mog_dfm.generate(
    objectives=objectives,
    weights=variant_probabilities
)
```

## Verification

To verify tree probabilities are correct:

```python
from tree_utils import load_tree_probabilities

variants = load_tree_probabilities("data/trees/hadsbm_tree.json")

# Check sum = 1.0
total = sum(v.probability for v in variants)
assert abs(total - 1.0) < 1e-6, f"Probabilities don't sum to 1.0: {total}"

# Check individual values
for v in variants:
    print(f"{v.name}: {v.probability:.4f}")
```

## References

- **Tree structure**: BranchSBM model from `tree_analysis` module
- **Split probabilities**: Computed from FastTree branch lengths in hadsbm_export.py
- **MOG-DFM weighting**: Multi-objective guided framework uses these weights for optimization
