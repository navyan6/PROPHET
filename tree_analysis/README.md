# Tree Analysis

Analysis and visualization of HIV phylogenetic trees.

## Structure

- **`src/`** - Core modules
  - `tree.py` - Tree data structures and operations
  - `phylogeny.py` - Phylogenetic analysis
  - `hadsbm_export.py` - Export functionality
  - `visualize_tree.py` - Tree visualization

- **`pipelines/`** - Processing pipelines
  - `pipeline_basic.py` - Basic pipeline
  - `pipeline_paths.py` - Path-based pipeline

- **`examples/`** - Example scripts

- **`docs/`** - Documentation

## Quick Start

```python
from tree_analysis.src import tree, phylogeny

# Build tree from sequences
tree = tree.build_tree_from_sequences("../data/sequences/hiv_sequences_aligned.fasta")

# Analyze phylogeny
phylogeny.analyze(tree)
```

## Data

Input and output data are stored in `/data`:
- `sequences/` - FASTA sequence files
- `trees/` - Tree files (Newick, JSON, PNG)
- `metadata/` - Associated metadata (CSV, etc.)
- `variants/` - Variant information (JSON)
