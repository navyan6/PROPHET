#!/usr/bin/env python3
"""
Build a phylogenetic tree on a sample of NCBI HIV protease sequences,
find a coherent clade, and save it as the held-out evaluation set.

The training sequences (var_0..var_58) come from a separate source and
have no overlap with this dataset, so no contamination check is needed.
"""
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio import Phylo, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

REPO = Path(__file__).resolve().parent.parent

FULL_FASTA  = REPO / "alignments/hiv_protease_aligned_full.fasta"
OUT_FASTA   = REPO / "data/pre_stage1_split/alignments/test/hiv_test_clade_holdout.fasta"
TREE_FASTA  = REPO / "data/pre_stage1_split/trees/hiv_full_subset_aligned.fasta"
OUT_TREE    = REPO / "data/pre_stage1_split/trees/hiv_full_subset_tree.nwk"

SAMPLE_SIZE = 1000
MIN_CLADE   = 100
MAX_CLADE   = 300
SEED        = 42
WT = "PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

random.seed(SEED)

# ── 1. Load and filter sequences ──────────────────────────────────────────────
print("Loading full alignment...")
all_seqs = []
seen = set()
for r in SeqIO.parse(FULL_FASTA, "fasta"):
    s = str(r.seq).replace("-", "")
    if len(s) != 99 or "X" in s or s in seen:
        continue
    seen.add(s)
    nid = r.id.split()[0].replace("|", "_").replace(".", "_")
    all_seqs.append((nid, s))

print(f"  {len(all_seqs)} unique valid sequences")

# ── 2. Filter to sequences with 5-20 edits from WT (exclude outliers) ────────
diverse = [(nid, s) for nid, s in all_seqs
           if 5 <= sum(a != b for a, b in zip(s, WT)) <= 25]
print(f"  {len(diverse)} with 5-25 edits from WT")

# ── 3. Sample and build tree ──────────────────────────────────────────────────
sampled = random.sample(diverse, min(SAMPLE_SIZE, len(diverse)))
print(f"\nBuilding tree on {len(sampled)} sequences...")

OUT_TREE.parent.mkdir(parents=True, exist_ok=True)
records = [SeqRecord(Seq(s), id=nid, description="") for nid, s in sampled]
SeqIO.write(records, TREE_FASTA, "fasta")

result = subprocess.run(
    ["FastTree", "-quiet", "-lg", str(TREE_FASTA)],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"FastTree error:\n{result.stderr}", file=sys.stderr)
    sys.exit(1)
OUT_TREE.write_text(result.stdout)
print(f"  Tree written → {OUT_TREE}")

# ── 4. Find a coherent clade ──────────────────────────────────────────────────
print(f"\nSearching for clade with {MIN_CLADE}–{MAX_CLADE} leaves...")
tree = Phylo.read(str(OUT_TREE), "newick")
tree.root_at_midpoint()

seq_lookup = {nid: s for nid, s in sampled}
candidates = []
for clade in tree.find_clades(order="level"):
    leaves = clade.get_terminals()
    n = len(leaves)
    if not (MIN_CLADE <= n <= MAX_CLADE):
        continue
    leaf_names = [c.name for c in leaves if c.name in seq_lookup]
    if len(leaf_names) < MIN_CLADE:
        continue
    leaf_seqs  = [seq_lookup[name] for name in leaf_names]
    edits = [sum(a != b for a, b in zip(s, WT)) for s in leaf_seqs]
    candidates.append((n, np.mean(edits), leaf_names, leaf_seqs))

if not candidates:
    print("ERROR: no suitable clade found.")
    sys.exit(1)

# Pick the clade whose mean edit distance is closest to 9.18 (paper's held-out)
candidates.sort(key=lambda x: abs(x[1] - 9.18))
best_n, best_edit, best_names, best_seqs = candidates[0]
edits = [sum(a != b for a, b in zip(s, WT)) for s in best_seqs]
print(f"  Selected clade: {best_n} leaves")
print(f"  Edit distance from WT: mean={np.mean(edits):.2f}, std={np.std(edits):.2f}, min={min(edits)}, max={max(edits)}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
out_records = [SeqRecord(Seq(s), id=nid, description="") for nid, s in zip(best_names, best_seqs)]
SeqIO.write(out_records, OUT_FASTA, "fasta")
print(f"\nSaved {len(out_records)} held-out sequences → {OUT_FASTA}")
