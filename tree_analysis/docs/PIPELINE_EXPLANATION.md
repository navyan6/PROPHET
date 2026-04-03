# HIV Phylogenetic Tree Building Pipeline

## Overview

This pipeline transforms HIV protease variant data into a phylogenetic tree with associated JSON data structure. The process has **4 main stages**:

1. **Variant Generation** (`tree.py`) — Extract protease sequences from UniProt JSON
2. **Tree Inference** (`phylogeny.py`) — Alignment + tree building (MAFFT → FastTree)
3. **Visualization** (`visualize_tree.py`) — Generate PNG and leaf path metadata
4. **HadSBM Export** (`hadsbm_export.py`) — Build JSON bundle with tree structure

---

## Stage 1: Variant Generation (`tree.py`)

### Purpose
Convert UniProt-style JSON containing natural variants into a FASTA file of HIV protease sequences.

### Input File Format
```json
{
  "sequence": "MNIFEMLRIDEGLGLQ....",
  "features": [
    {
      "type": "VARIANT",
      "begin": 30,    
      "end": 30,
      "wildType": "M",
      "alternativeSequence": "I"
    },
    ...
  ]
}
```

### Process

```
hiv-variants.json
       ↓
  [read_wildtype_polyprotein]
       ↓
  Extract full 100+ aa polyprotein (or from wildtype_fasta if provided)
       ↓
  [protease_start_index]
       ↓
  Find protease start marker ("PQVTLWQR" or "PQITLWQR") in polyprotein
       ↓
  For EACH variant feature:
    - Check if position is within protease window (99 aa after start)
    - Verify single amino-acid substitution
    - Create new 99-letter sequence: WT protease + variant substitution
    - Write as: ">var_<index>\n<sequence>\n"
       ↓
  hiv_sequences.fasta (one variant per line)
```

### Key Functions

| Function | Role |
|----------|------|
| `read_wildtype_polyprotein()` | Load reference polyprotein (JSON or FASTA) |
| `protease_start_index()` | Locate protease within polyprotein using known markers |
| `generate_variants_from_json()` | Main function: extract all valid variants |

### Output Example
```fasta
>var_0
PQVTLWQRPLVTIKIGGQLKEALLDTG...  (99 aa with 1 substitution from WT)
>var_1
PQVTLWQRPLVTIKIGGQLKEALLDNG...  (different substitution)
...
```

---

## Stage 2: Tree Inference (`phylogeny.py`)

### Purpose
Build a Newick phylogenetic tree from protease sequences using MSA and maximum-likelihood inference.

### Process

```
hiv_sequences.fasta
       ↓
  [run_phylogeny] checks tools: mafft, FastTree
       ↓
  MAFFT Alignment
  ├─ Command: mafft --auto hiv_sequences.fasta
  ├─ Output: Aligned FASTA with gaps (-) to match columns
  └─ Result:  hiv_sequences_aligned.fasta
       ↓
  FastTree Tree Inference
  ├─ Command: FastTree -lg hiv_sequences_aligned.fasta
  ├─ Uses: Log-likelihood with gamma distribution model (-lg)
  ├─ Output: Newick format with branch lengths (substitutions/site)
  └─ Result: hiv_tree.nwk
```

### Branch Lengths & Confidence
- **Branch lengths**: Substitutions per site (from evolutionary distance)
- **Confidence**: SH-like local support values (≈ 0–1, NOT bootstrap %)
  - Written as internal node labels in Newick format
  - Example: `(A:0.1,B:0.05)0.95:0.01;` means 95% confidence at that split

### Dependencies
- **MAFFT**: Multiple sequence alignment tool
- **FastTree**: Fast approximate maximum-likelihood tree builder

### Output Format (Newick)
```
(var_0:0.012,var_1:0.008)0.87:0.002;
```
- `var_0, var_1`: Leaf names (sequence IDs)
- `0.012, 0.008`: Branch lengths from common ancestor
- `0.87`: SH-like support at that split
- Tree is **rooted** at the root, leaves are terminal nodes

---

## Stage 3: Visualization (`visualize_tree.py`)

### Purpose
Generate PNG visualization and extract per-leaf metadata.

### Outputs

#### 1. PNG/SVG Figure
```
[draw_tree_to_png]
  ├─ Reads: Newick tree
  ├─ Uses: matplotlib via Bio.Phylo
  ├─ Shows: Node labels (SH-like supports), edge lengths
  └─ Result: hiv_tree.png (size: 14" × 20" @ 120 DPI)
```

#### 2. CSV: Leaf Path Metadata (`leaf_paths.csv`)
```
leaf,n_supports_on_path,supports_root_to_leaf,sum_branch_length,branch_lengths_root_to_leaf
var_0,3,0.95;0.88;0.72,0.045000,"0.010000;0.012000;0.023000"
var_1,3,0.95;0.88;0.76,0.038000,"0.010000;0.012000;0.016000"
```

**Columns**:
- `leaf`: Sequence identifier
- `n_supports_on_path`: Number of internal nodes between root and this leaf
- `supports_root_to_leaf`: SH-like support values separated by semicolons (root → leaf)
- `sum_branch_length`: Total distance from root to this leaf
- `branch_lengths_root_to_leaf`: Individual branch lengths (root → leaf)

**Purpose**: Track evolutionary distance and confidence per sequence variant.

---

## Stage 4: HadSBM Export (`hadsbm_export.py`)

### Purpose
Build a JSON data structure containing the complete tree topology and sequence information for BranchSBM inference.

### Process Architecture

```
hiv_tree.nwk (Newick)
hiv_sequences.fasta (FASTA)
hiv-variants.json (WT reference)
       ↓
  [build_hadsbm_bundle]
       ├─ [Preorder Traversal]
       │  └─→ Traverse tree root-to-leaves, assign node indices
       │
       ├─ [Node Mapping]
       │  ├─ Create index_of: {clade_id} → node_index
       │  └─ Track parent_of: [parent indices]
       │
       ├─ [Normalize Time]
       │  ├─ Find deepest root-to-leaf path
       │  └─ Convert branch lengths to tau ∈ [0,1]
       │
       ├─ [Build Adjacency]
       │  ├─ Create n×n matrix G (symmetric, 1 = edge)
       │  └─ G[i,j] = G[j,i] = 1 if edge between nodes
       │
       ├─ [Extract Splits]
       │  └─ For each internal node with 2 children:
       │     - Record parent & child indices
       │     - Store normalized time tau
       │     - Compute p_left, p_right (split weights)
       │     - Get SH-like support (confidence)
       │
       ├─ [Extract Leaves]
       │  └─ For each terminal node:
       │     - Store node index
       │     - Store sequence ID (leaf_id)
       │     - Store amino acid sequence
       │
       └─ [Assemble JSON]
          └─→ hadsbm_tree.json
```

### Key Data Transformations

#### 1. Preorder Node Indexing
Traverse tree (root first, then children) and assign indices 0, 1, 2, ...
```
     (0)                Root first
    /   \
  (1)   (2)            Children next
  / \
(3) (4)                Leaves at end
```

#### 2. Time Normalization
- **Raw**: Branch lengths accumulate from root to tips
- **Normalized (tau)**: tau = (depth_to_node) / (max_root_to_tip_depth)
  - Result: tau ∈ [0, 1] interval (time flows from 0 → 1)

#### 3. Split Weights (p_left, p_right)
Two modes:
- **"length"** (default): `p_left = branch_len_left / (branch_len_left + branch_len_right)`
  - Heavier branches get higher probability
- **"uniform"**: `p_left = p_right = 0.5`
  - Symmetric split regardless of branch length

### Output JSON Structure

```json
{
  "format": "hadsbm_tree_v1",
  "description": {
    "G": "Symmetric adjacency; node order = preorder",
    "tau": "Split times in [0,1]",
    "p_k": "Binary split weights; mode=length",
    "pi_1_k": "Leaf sequences",
    "x_WT": "Wild-type anchor"
  },
  "x_WT": "PQVTLWQRPLVTIK...",  // 99-aa protease WT
  "n_nodes": 150,                 // Total nodes (internal + leaves)
  "n_leaves": 93,                 // Terminal nodes
  "node_order": "preorder from Newick root",
  
  "adjacency_G": [
    [0, 1, 0, 0, ...],
    [1, 0, 1, 1, ...],
    ...
  ],
  
  "parent_index": [-1, 0, 0, 1, 1, 2, ...],  // -1 = root
  
  "splits": [
    {
      "parent_index": 0,
      "left_child_index": 1,
      "right_child_index": 2,
      "time_tau": 0.95,           // Split occurs at 95% of tree depth
      "p_left": 0.58,             // 58% of mass goes left
      "p_right": 0.42,            // 42% goes right
      "sh_support": 0.87,         // 87% SH-like confidence
      "branch_len_left": 0.012,
      "branch_len_right": 0.008
    },
    ...
  ],
  
  "leaf_endpoints_pi": [
    {
      "node_index": 145,
      "leaf_id": "var_0",
      "sequence": "PQVTLWQRPLV..."      // Full 99-aa sequence
    },
    ...
  ],
  
  "leaf_ids_in_order": ["var_0", "var_1", ...]
}
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `SplitEvent` | One binary split (parent + 2 children, time, weights, confidence) |
| `LeafEndpoint` | Terminal node (index, ID, sequence) |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: hiv-variants.json (UniProt format)                            │
│        wildtype.fasta (optional WT FASTA)                            │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
                        [tree.py]
                   Variant Extraction
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ hiv_sequences.fasta (multiple-FASTA, 1 variant per sequence)        │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
                     [phylogeny.py]
              MAFFT + FastTree Tree Inference
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ hiv_sequences_aligned.fasta (aligned variant sequences)              │
│ hiv_tree.nwk (Newick format tree)                                   │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
                   [visualize_tree.py]
            ┌──────────────────────────────┐
            ├─→ hiv_tree.png (visualization)
            └─→ leaf_paths.csv (per-leaf metadata)
                               ↓
                    [hadsbm_export.py]
           Build JSON Tree Structure for BranchSBM
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ hadsbm_tree.json (complete tree data structure)                      │
│ ├─ Topology: adjacency matrix, parent links                          │
│ ├─ Splits: parent/children, normalized times, split weights         │
│ ├─ Leaves: node indices, sequence IDs, sequences                    │
│ └─ WT anchor: reference protease sequence                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How hadsbm_export.py Does Its Work (Detailed)

### Step 1: Parse Newick and Build Node List
```python
nodes = list_nodes_preorder(root)  # Traverse root-first
```
- Input: Bio.Phylo.Tree object (from Newick file)
- Output: List of clades in preorder (parent before children)
- Enables: Mapping each clade to a numerical index

### Step 2: Index Mapping & Adjacency
```python
for index, clade in enumerate(nodes):
    index_of[id(clade)] = index  # Map clade object → index
```
- Build two dictionaries:
  - `index_of`: Maps clade Python objects to indices 0, 1, 2, ...
  - `parent_of`: Tracks parent index for each node

### Step 3: Build Adjacency Matrix
```python
adjacency[i][j] = 1  if edge exists between nodes i and j
```
- Undirected graph representation (symmetric matrix)
- Row i, Column j = 1 means edge from node i to node j

### Step 4: Extract and Normalize Splits
For each internal node with exactly 2 children:

```python
depth_at_split = depth_from_root_to_clade(root, clade)  # Branch units
tau = depth_at_split / deepest_path                     # Normalize to [0,1]
```

- **Time**: Accumulated branch length → normalized proportion
- **Weights**: Branch lengths → split probability
  - Length-based: p_left = left_len / (left_len + right_len)
  - Uniform: p_left = p_right = 0.5
- **Confidence**: Extract SH-like support from internal node labels

### Step 5: Extract Leaves
```python
for clade in nodes:
    if clade.is_terminal():
        leaf_id = clade.name
        sequence = fasta_by_id[leaf_id]  # Lookup from FASTA
        leaf_list.append(LeafEndpoint(...))
```

- Match leaf names to sequences from FASTA file
- Raise error if leaf in tree but not in FASTA (data mismatch)

### Step 6: Assemble JSON
Combine all components into a single dictionary:
- Metadata & format version
- Tree topology (adjacency, parent links)
- Splits with times, weights, supports
- Leaf sequences
- Wild-type anchor sequence

---

## Running the Pipeline

### Option 1: Full Automated Pipeline
```bash
cd tree_analysis
python pipelines/pipeline_basic.py \
  --skip-fasta \           # Don't rebuild FASTA (if already exists)
  --prob-mode length \     # Use branch-length-weighted splits (default)
  --ascii-tree             # Print ASCII tree to terminal
```

### Option 2: Step-by-Step
```bash
# 1. Generate variants from JSON
python src/tree.py --json hiv-variants.json --out hiv_sequences.fasta

# 2. Align + build tree
python src/phylogeny.py

# 3. Visualize + export metrics
python src/visualize_tree.py --ascii

# 4. Export to HadSBM JSON
python src/hadsbm_export.py --prob-mode length
```

### Option 3: Custom Paths
```bash
python src/hadsbm_export.py \
  --nwk /path/to/custom_tree.nwk \
  --fasta /path/to/custom.fasta \
  --variants-json /path/to/custom_variants.json \
  --wt-fasta /path/to/wildtype.fasta \
  --out /path/to/output.json \
  --prob-mode uniform
```

---

## Key Design Decisions

### Why Preorder Indexing?
- **Preorder**: Visit parent before children
- Ensures parent indices are always < child indices in dependency graph
- Simplifies tree traversal algorithms

### Why Normalize Time to [0,1]?
- Different datasets have different absolute branch lengths
- Normalized tau makes models generalizable across trees
- BranchSBM uses time as continuous variable for variational inference

### Why Two Split Weight Modes?
- **"length"**: Incorporates branch length uncertainty into prior
- **"uniform"**: Ignores branch lengths, symmetric prior
- Trade-off: Information vs. robustness to alignment uncertainty

### Why Separate WT Anchor?
- Some analyses use WT as reference point
- BranchSBM can condition on known WT sequence
- Enables comparison of variants against baseline

---

## Output Files Summary

| File | Format | Purpose |
|------|--------|---------|
| `hiv_sequences.fasta` | Multi-FASTA | One 99-aa protease variant per sequence |
| `hiv_sequences_aligned.fasta` | Aligned FASTA | MAFFT alignment output (with gaps) |
| `hiv_tree.nwk` | Newick | Phylogenetic tree topology + branch lengths |
| `hiv_tree.png` | PNG image | Visualization with confidence values |
| `leaf_paths.csv` | CSV table | Per-leaf evolutionary metrics |
| `hadsbm_tree.json` | JSON | Complete tree data for BranchSBM analysis |

---

## Troubleshooting

### Missing Tools
- **MAFFT**: `brew install mafft` or `conda install -c bioconda mafft`
- **FastTree**: `conda install -c bioconda fasttree`

### Leaf Mismatch Error
```
KeyError: Leaf 'var_0' is in the tree but not in the FASTA file
```
- Cause: Tree was built from a different FASTA file
- Solution: Regenerate tree from same FASTA using `phylogeny.py`

### Invalid Protease Motif
```
ValueError: Could not find protease start motif
```
- Cause: WT sequence doesn't contain known protease markers
- Solution: Provide correct `--wt-fasta` or update `PROTEASE_START_MARKERS`

---

## References

- **Bio.Phylo**: https://biopython.org/wiki/Phylo
- **Newick Format**: http://marvin.cs.uchicago.edu/phylogenetics/phlip/newick.html
- **MAFFT**: https://mafft.cbrc.jp/alignment/software/
- **FastTree**: http://www.microbesonline.org/fasttree/
- **BranchSBM**: [Local reference paper/model]
